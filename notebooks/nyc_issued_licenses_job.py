# pylint: disable=import-error,undefined-variable,broad-exception-caught
"""NYC Issued Licenses Job.

Fetches new NYC business licenses from the Socrata API since the last recorded
creation date and merges them into a Delta Lake table. Handles schema
inference from existing data and manages incremental updates efficiently.
"""

from datetime import datetime
from sodapy import Socrata
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from delta.tables import DeltaTable

APP_TOKEN = dbutils.secrets.get("biz-survival", "nyc_app_token")
TABLE_PATH = "data_515.default.issued_licenses"
COLUMNS_STR = "*"
ISSUED_LICENSE_SCHEMA = None

try:
    df_existing = spark.table(TABLE_PATH)
    columns_list = df_existing.columns
    COLUMNS_STR = ','.join(columns_list)
    ISSUED_LICENSE_SCHEMA = StructType(
        [StructField(c, StringType(), True) for c in columns_list]
    )
    last_created_date_str = df_existing.agg(F.max('license_creation_date')).collect()[0][0]
    last_created_date = last_created_date_str.strftime("%Y-%m-%dT00:00:00.000")
except Exception:
    last_created_date = datetime.today().replace(day=1).strftime("%Y-%m-%dT00:00:00.000")

print("Last Created Date: ", last_created_date)

client = Socrata("data.cityofnewyork.us", APP_TOKEN)
SOQL_QUERY = (
    f"SELECT {COLUMNS_STR} "
    f"WHERE license_creation_date > '{last_created_date}' "
    "LIMIT 200000"
)
try:
    results = client.get("w7w3-xahh", query=SOQL_QUERY)
finally:
    client.close()

if results:
    if ISSUED_LICENSE_SCHEMA is not None:
        df_new_data = spark.createDataFrame(results, schema=ISSUED_LICENSE_SCHEMA)
    else:
        df_new_data = spark.createDataFrame(results)
    df_new_data = (
        df_new_data
        .withColumn("license_creation_date", F.to_timestamp("license_creation_date"))
        .withColumn("lic_expir_dd", F.to_timestamp("lic_expir_dd"))
    )
    print("New Data Count:", df_new_data.count())

    DeltaTable.forName(spark, TABLE_PATH) \
        .alias("target") \
        .merge(
            df_new_data.alias("source"),
            "target.license_nbr = source.license_nbr"
        ) \
        .whenMatchedUpdateAll() \
        .whenNotMatchedInsertAll() \
        .execute()
    print("Merge completed")
else:
    print("No new data")
