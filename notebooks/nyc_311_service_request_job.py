# pylint: disable=import-error,undefined-variable

"""NYC 311 Service Request Job.

Fetches new service requests from the NYC 311 API since the last recorded
creation date and merges them into a Delta Lake table. Handles schema
inference from existing data and manages incremental updates efficiently.
"""

from datetime import datetime
from sodapy import Socrata
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from delta.tables import DeltaTable

APP_TOKEN = dbutils.secrets.get("biz-survival", "nyc_app_token")
TABLE_PATH = "data_515.default.311_service_requests"
COLUMNS_STR = "*"
NYC_311_SCHEMA = None
FETCH_LIMIT = 200000


try:
    df_existing = spark.table(TABLE_PATH)
    columns_list = df_existing.columns
    COLUMNS_STR = ",".join(columns_list)
    NYC_311_SCHEMA = StructType(
        [StructField(c, StringType(), True) for c in columns_list]
    )
    last_created_date_raw = df_existing.agg(F.max("created_date")).collect()[0][0]
    if last_created_date_raw is None:
        raise TypeError("No existing data found")
    if isinstance(last_created_date_raw, str):
        last_created_date = last_created_date_raw[:10] + "T00:00:00.000"
    else:
        last_created_date = last_created_date_raw.strftime("%Y-%m-%dT00:00:00.000")
except (TypeError, AttributeError, ValueError):
    last_created_date = (
        datetime.today().replace(day=1).strftime("%Y-%m-%dT00:00:00.000")
    )

print("Last Created Date:", last_created_date)

SOQL_QUERY = (
    f"SELECT {COLUMNS_STR} "
    f"WHERE created_date > '{last_created_date}' "
    f"LIMIT {FETCH_LIMIT}"
)

client = Socrata("data.cityofnewyork.us", APP_TOKEN)
try:
    results = client.get("erm2-nwe9", query=SOQL_QUERY)
finally:
    client.close()

if results:
    if NYC_311_SCHEMA is not None:
        normalized = [
            {column: row.get(column, None) for column in columns_list}
            for row in results
        ]
        df_new_data = spark.createDataFrame(normalized, schema=NYC_311_SCHEMA)
    else:
        df_new_data = spark.createDataFrame(results)
    df_new_data = (
        df_new_data
        .withColumn("created_date", F.to_timestamp("created_date"))
        .withColumn("closed_date", F.to_timestamp("closed_date"))
        .withColumn(
            "resolution_action_updated_date",
            F.to_timestamp("resolution_action_updated_date"),
        )
    )
    print("New Data Count:", df_new_data.count())

    DeltaTable.forName(spark, TABLE_PATH).alias("target").merge(
        df_new_data.alias("source"), "target.unique_key = source.unique_key"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    print("Merge completed")
else:
    print("No new data")
