import sys
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import col, initcap, lit
from delta.tables import DeltaTable
from awsglue.transforms import EvaluateDataQuality
from awsglue.job import Job
from awsglueml.transforms import EntityDetector
from awsglue.transforms import *
from awsglue.dynamicframe import DynamicFrame
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

DEFAULT_DATA_QUALITY_RULESET = """
    Rules = [
        ColumnCount > 0
    ]
"""

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

spark.conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
spark.conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

logger.info("Reading data from Glue catalog")

customer_df = glueContext.create_dynamic_frame.from_catalog(
    database="your_db", table_name="customer", transformation_ctx="customer_df").toDF()

orders_df = glueContext.create_dynamic_frame.from_catalog(
    database="your_db", table_name="orders").toDF()

logger.info("Applying transformations")

new_orders_df = orders_df.withColumn("first_name", initcap(col("first_name")))
output_path_orders = "s3://your-bucket/processed/orders/"
output_path_customer = "s3://your-bucket/processed/customer/"

logger.info("Writing data in Delta format")

new_orders_df.write.format("delta").mode("append").save(output_path_orders)

logger.info("orders Delta write completed successfully.")

entity_detector = EntityDetector()
classified_map = entity_detector.classify_columns(customer_df, ["PHONE_NUMBER", "USA_ATIN", "USA_PASSPORT_NUMBER", "USA_PTIN", "USA_SSN", "USA_ITIN", "BANK_ACCOUNT"], 1.0, 0.1, "HIGH")

def maskDf(df, keys):
    if not keys:
        return df
    df_to_mask = df.toDF()
    for key in keys:
        df_to_mask = df_to_mask.withColumn(key, lit("*******"))
    return DynamicFrame.fromDF(df_to_mask, glueContext, "updated_masked_df")

DetectSensitiveData_node1747203820950 = maskDf(customer_df, list(classified_map.keys()))

# Script generated for node Amazon S3
EvaluateDataQuality().process_rows(frame=DetectSensitiveData_node1747203820950, ruleset=DEFAULT_DATA_QUALITY_RULESET, publishing_options={"dataQualityEvaluationContext": "EvaluateDataQuality_node1747225488950", "enableDataQualityResultsPublishing": True}, additional_options={"dataQualityResultsPublishing.strategy": "BEST_EFFORT", "observations.scope": "ALL"})
additional_options = {"path": "s3://krkdata/peocessed/", "write.parquet.compression-codec": "snappy"}
AmazonS3_node1747226815204_df = DetectSensitiveData_node1747203820950.toDF()
AmazonS3_node1747226815204_df.write.format("delta").options(**additional_options).mode("append").save(output_path_customer)

job.commit()