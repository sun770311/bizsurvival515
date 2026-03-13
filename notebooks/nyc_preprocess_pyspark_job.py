# pylint: disable=import-error,undefined-variable,too-many-locals

"""Databricks PySpark preprocessing job for business survival features.
Inputs (Delta tables):
- data_515.default.issued_licenses
- data_515.default.311_service_requests

Output (Delta table):
- data_515.default.business_survival_preprocessed
"""

from functools import reduce

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

LICENSES_TABLE = "data_515.default.issued_licenses"
SERVICE_REQS_TABLE = "data_515.default.311_service_requests"
OUTPUT_TABLE = "data_515.default.business_survival_preprocessed"
LOCATION_K = 25


def parse_timestamp(col_expr):
    """Timestamp parser with explicit cast for safety."""
    return F.to_timestamp(col_expr)


def sanitize_feature_expr(col_expr, prefix: str):
    """Spark expression equivalent of sanitize_feature_name()."""
    cleaned = F.lower(F.coalesce(col_expr.cast("string"), F.lit("unknown")))
    cleaned = F.regexp_replace(cleaned, "&", "and")
    cleaned = F.regexp_replace(cleaned, r"[^a-z0-9_]+", "_")
    cleaned = F.regexp_replace(cleaned, r"_+", "_")
    cleaned = F.regexp_replace(cleaned, r"^_+|_+$", "")
    return F.concat(F.lit(f"{prefix}_"), cleaned)


def make_unique_column_names(columns: list[str]) -> list[str]:
    """Ensure all output column names are unique."""
    seen: dict[str, int] = {}
    unique_columns: list[str] = []

    for col_name in columns:
        if col_name not in seen:
            seen[col_name] = 1
            unique_columns.append(col_name)
        else:
            new_col = f"{col_name}_{seen[col_name]}"
            seen[col_name] += 1
            unique_columns.append(new_col)

    return unique_columns


def clean_licenses_spark(df: DataFrame) -> DataFrame:
    """Minimal cleaning for issued licenses with canonical schema."""

    cleaned = df.select(
        F.trim(F.col("business_unique_id").cast("string")).alias("business_id"),
        parse_timestamp(F.col("license_creation_date")).alias("initial_issuance_date"),
        parse_timestamp(F.col("lic_expir_dd")).alias("expiration_date"),
        F.when(
            F.trim(F.col("business_category").cast("string")) == "", F.lit("unknown")
        )
        .otherwise(F.trim(F.col("business_category").cast("string")))
        .alias("business_category"),
        F.col("latitude").cast("double").alias("business_latitude"),
        F.col("longitude").cast("double").alias("business_longitude"),
    )

    return cleaned.filter(
        (F.col("business_id").isNotNull())
        & (F.col("business_id") != "")
        & F.col("initial_issuance_date").isNotNull()
        & (
            F.col("expiration_date").isNull()
            | (F.col("expiration_date") >= F.col("initial_issuance_date"))
        )
    )


def clean_service_requests_spark(df: DataFrame) -> DataFrame:
    """Minimal cleaning for 311 requests with canonical schema."""

    cleaned = df.select(
        F.trim(F.col("unique_key").cast("string")).alias("unique_key"),
        parse_timestamp(F.col("created_date")).alias("created_date"),
        F.trim(F.col("complaint_type").cast("string")).alias("problem_type_raw"),
        F.col("latitude").cast("double").alias("latitude"),
        F.col("longitude").cast("double").alias("longitude"),
    )

    return cleaned.filter(
        (F.col("unique_key").isNotNull())
        & (F.col("unique_key") != "")
        & F.col("created_date").isNotNull()
        & F.col("problem_type_raw").isNotNull()
        & (F.col("problem_type_raw") != "")
    ).withColumn("month", F.to_timestamp(F.date_trunc("month", F.col("created_date"))))


def build_base_panel_spark(licenses: DataFrame) -> DataFrame:
    """Generate business-month panel from license lifetimes."""
    panel = (
        licenses.withColumn(
            "start_m", F.to_date(F.date_trunc("month", F.col("initial_issuance_date")))
        )
        .withColumn(
            "end_m",
            F.to_date(
                F.date_trunc(
                    "month", F.coalesce(F.col("expiration_date"), F.current_timestamp())
                )
            ),
        )
        .withColumn(
            "month",
            F.explode(
                F.sequence(F.col("start_m"), F.col("end_m"), F.expr("interval 1 month"))
            ),
        )
        .withColumn("month", F.to_timestamp(F.col("month")))
        .select(
            "business_id",
            "month",
            "business_latitude",
            "business_longitude",
            "business_category",
        )
    )

    panel_agg = panel.groupBy("business_id", "month").agg(
        F.count("business_category").alias("active_license_count"),
        F.first("business_latitude", ignorenulls=True).alias("business_latitude"),
        F.first("business_longitude", ignorenulls=True).alias("business_longitude"),
        F.collect_list("business_category").alias("category_list"),
    )

    first_license_month = F.min("month").over(Window.partitionBy("business_id"))
    return panel_agg.withColumn(
        "months_since_first_license",
        F.floor(F.months_between(F.col("month"), first_license_month)).cast("int"),
    ).withColumn("open", F.lit(1))


def add_location_clusters_spark(panel: DataFrame, location_k: int) -> DataFrame:
    """Add KMeans location clusters and cluster centers."""
    panel_base = (
        panel.withColumn("location_cluster", F.lit(0).cast("int"))
        .withColumn("location_cluster_lat", F.lit(0.0).cast("double"))
        .withColumn("location_cluster_lng", F.lit(0.0).cast("double"))
    )

    coords = panel_base.filter(
        F.col("business_latitude").isNotNull() & F.col("business_longitude").isNotNull()
    )

    coords_count = coords.count()
    if coords_count < location_k or location_k <= 0:
        return panel_base

    keyed = panel_base.withColumn("_row_id", F.monotonically_increasing_id())
    coords_keyed = keyed.filter(
        F.col("business_latitude").isNotNull() & F.col("business_longitude").isNotNull()
    )

    assembler = VectorAssembler(
        inputCols=["business_latitude", "business_longitude"],
        outputCol="features",
    )
    features_df = assembler.transform(coords_keyed)

    model = KMeans(
        k=location_k, seed=42, featuresCol="features", predictionCol="prediction"
    ).fit(features_df)
    scored = model.transform(features_df).select(
        "_row_id", F.col("prediction").cast("int").alias("location_cluster")
    )

    centers = model.clusterCenters()
    lat_pairs = []
    lng_pairs = []
    for i, center in enumerate(centers):
        lat_pairs.extend([F.lit(int(i)), F.lit(float(center[0]))])
        lng_pairs.extend([F.lit(int(i)), F.lit(float(center[1]))])

    lat_map = F.create_map(*lat_pairs)
    lng_map = F.create_map(*lng_pairs)

    enriched = (
        keyed.join(scored, on="_row_id", how="left")
        .withColumn(
            "location_cluster",
            F.coalesce(F.col("location_cluster"), F.lit(0)).cast("int"),
        )
        .withColumn(
            "location_cluster_lat",
            F.coalesce(
                F.element_at(lat_map, F.col("location_cluster")), F.lit(0.0)
            ).cast("double"),
        )
        .withColumn(
            "location_cluster_lng",
            F.coalesce(
                F.element_at(lng_map, F.col("location_cluster")), F.lit(0.0)
            ).cast("double"),
        )
        .drop("_row_id")
    )

    return enriched


def add_category_dummies_spark(panel: DataFrame) -> DataFrame:
    """Pivot categories into dummy flags and attach to panel."""
    categories = (
        panel.select(
            "business_id", "month", F.explode_outer("category_list").alias("category")
        )
        .withColumn(
            "category_feature",
            sanitize_feature_expr(F.col("category"), "business_category"),
        )
        .withColumn("value", F.lit(1))
    )

    category_features = [
        row[0]
        for row in categories.select("category_feature").distinct().collect()
        if row[0] is not None and row[0] != "business_category_"
    ]

    if not category_features:
        return panel.withColumn("business_category_sum", F.lit(0))

    pivoted = (
        categories.groupBy("business_id", "month")
        .pivot("category_feature", category_features)
        .agg(F.max("value"))
        .fillna(0)
    )

    combined = panel.join(pivoted, on=["business_id", "month"], how="left")
    for col_name in category_features:
        combined = combined.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0)))

    return combined.withColumn(
        "business_category_sum",
        reduce(lambda a, b: a + b, [F.col(c) for c in category_features]),
    )


def process_complaints_spark(reqs: DataFrame) -> tuple[DataFrame, list[str]]:
    """Pivot 311 requests by month into complaint feature columns."""
    reqs_agg = reqs.groupBy("month", "problem_type_raw").agg(
        F.count(F.lit(1)).alias("complaint_count")
    )

    reqs_features = reqs_agg.withColumn(
        "complaint_feature",
        sanitize_feature_expr(F.col("problem_type_raw"), "complaint_type"),
    )

    complaint_cols = [
        row[0]
        for row in reqs_features.select("complaint_feature").distinct().collect()
        if row[0] is not None and row[0] != "complaint_type_"
    ]

    if not complaint_cols:
        return reqs_features.select("month").distinct(), []

    pivoted = (
        reqs_features.groupBy("month")
        .pivot("complaint_feature", complaint_cols)
        .agg(F.sum("complaint_count"))
        .fillna(0)
    )

    return pivoted, complaint_cols


def build_joined_dataset_spark(
    licenses_table: str,
    service_reqs_table: str,
    location_k: int,
) -> DataFrame:
    """Build final joined business panel dataset in Spark."""
    licenses_raw = spark.table(licenses_table)
    reqs_raw = spark.table(service_reqs_table)

    licenses = clean_licenses_spark(licenses_raw)
    reqs = clean_service_requests_spark(reqs_raw)

    panel_agg = build_base_panel_spark(licenses)
    panel_with_clusters = add_location_clusters_spark(panel_agg, location_k)
    panel_with_cats = add_category_dummies_spark(panel_with_clusters)

    complaint_pivot, complaint_cols = process_complaints_spark(reqs)

    final_panel = panel_with_cats.join(complaint_pivot, on="month", how="left")
    for col_name in complaint_cols:
        final_panel = final_panel.withColumn(
            col_name, F.coalesce(F.col(col_name), F.lit(0))
        )

    if complaint_cols:
        complaint_sum_expr = reduce(
            lambda a, b: a + b, [F.col(c) for c in complaint_cols]
        )
        final_panel = final_panel.withColumn("complaint_sum", complaint_sum_expr)
    else:
        final_panel = final_panel.withColumn("complaint_sum", F.lit(0))

    final_panel = final_panel.withColumn("total_311", F.col("complaint_sum"))

    # Keep parity with pandas pipeline behavior.
    final_panel = final_panel.drop("category_list")

    # Defensive: ensure uniqueness in case of accidental duplicate names.
    unique_columns = make_unique_column_names(final_panel.columns)
    return final_panel.toDF(*unique_columns)


final_df = build_joined_dataset_spark(
    licenses_table=LICENSES_TABLE,
    service_reqs_table=SERVICE_REQS_TABLE,
    location_k=LOCATION_K,
)

(
    final_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Saved preprocessed dataset to {OUTPUT_TABLE}")
print(f"Row count: {final_df.count()}")
