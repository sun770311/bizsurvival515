# Databricks notebook source
# pylint: disable=import-error,undefined-variable,broad-exception-caught
from sodapy import Socrata
from pyspark.sql import functions as F
from delta.tables import DeltaTable
from datetime import datetime
from pyspark.sql.types import StructType, StructField, StringType

APP_TOKEN = dbutils.secrets.get("biz-survival", "nyc_app_token")
lake_table_path = "data_515.default.issued_licenses"
columns_str = "*"
nyc_311_schema = None

try:
    df_existing = spark.table(lake_table_path)
    columns_list = df_existing.columns
    columns_str = ','.join(columns_list)
    nyc_311_schema = StructType([StructField(c, StringType(), True) for c in columns_list])
    last_created_date_str = df_existing.agg(F.max('license_creation_date')).collect()[0][0]
    last_created_date = last_created_date_str.strftime("%Y-%m-%dT00:00:00.000")
except Exception:
    last_created_date = datetime.today().replace(day=1).strftime("%Y-%m-%dT00:00:00.000")

print("Last Created Date: ", last_created_date)

client = Socrata("data.cityofnewyork.us", APP_TOKEN)
soql_query = f"SELECT {columns_str} WHERE license_creation_date > '{last_created_date}' limit 200000"
try:
    results = client.get("w7w3-xahh", query=soql_query)
finally:
    client.close()

if results:
    if nyc_311_schema is not None:
        df_new_data = spark.createDataFrame(results, schema=nyc_311_schema)
    else:
        df_new_data = spark.createDataFrame(results)
    df_new_data = df_new_data.withColumn("license_creation_date", F.to_timestamp("license_creation_date"))\
                            .withColumn("lic_expir_dd", F.to_timestamp("lic_expir_dd"))
    print("New Data Count:", df_new_data.count())

    DeltaTable.forName(spark, lake_table_path) \
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
