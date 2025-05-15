import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True, help='Environment (dev/prod)')
parser.add_argument('--base_path', required=True, help='Base S3 path')
args = parser.parse_args()

ENV = args.env
BASE_PATH = args.base_path 

CUSTOMER_PATH = f"{BASE_PATH}/curated/customer/"
ORDERS_PATH = f"{BASE_PATH}/curated/orders/"
OUTPUT_PATH = f"{BASE_PATH}/final/aggregated_users/{ENV}/"

spark = SparkSession.builder \
    .appName("EMR Delta Aggregation Job") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

logger.info("Reading Delta tables from curated layer")


customer_df = spark.read.format("delta").load(CUSTOMER_PATH)
orders_df = spark.read.format("delta").load(ORDERS_PATH)

logger.info("Joining customer and orders")

joined_df = orders_df.join(customer_df, on="customer_id", how="inner")

user_orders = joined_df.groupBy("customer_id", "customer_name") \
    .agg(count("order_id").alias("total_products_ordered"))

top_20_customers = user_orders.orderBy(desc("total_products_ordered")).limit(20)

logger.info(f"Writing aggregated data to {OUTPUT_PATH}")

top_20_customers.write \
    .format("delta") \
    .mode("overwrite") \
    .save(OUTPUT_PATH)

logger.info("Delta aggregation job completed successfully.")