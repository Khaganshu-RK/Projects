import sys
import boto3
import logging
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.dynamicframe import DynamicFrame
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.utils import getResolvedOptions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

sc = SparkContext()
glueContext = GlueContext(sc)
job = Job(glueContext)


args = getResolvedOptions(sys.argv, ['TABLE_NAME', 'DATABASE_NAME', 'SNS_TOPIC_ARN', 'MODE'])
table_name = args['TABLE_NAME']
database_name = args['DATABASE_NAME']
sns_topic_arn = args['SNS_TOPIC_ARN']
mode = args['MODE']

glue = boto3.client('glue')
sns = boto3.client('sns')

versions = glue.get_table_versions(DatabaseName=database_name, TableName=table_name)['TableVersions']
sorted_versions = sorted(versions, key=lambda v: int(v['VersionId']))

if len(sorted_versions) > 1:
    latest = sorted_versions[-1]['Table']
    previous = sorted_versions[-2]['Table']

    latest_cols_list = latest['StorageDescriptor']['Columns']
    previous_cols_list = previous['StorageDescriptor']['Columns']

    latest_cols_dict = {col['Name']: col['Type'] for col in latest_cols_list}
    previous_cols_dict = {col['Name']: col['Type'] for col in previous_cols_list}

    changes = []

    for col in previous_cols_dict:
        if col not in latest_cols_dict:
            changes.append(f"[REMOVED] Column '{col}' was removed.")
        elif latest_cols_dict[col] != previous_cols_dict[col]:
            changes.append(f"[CHANGED] Column '{col}' type changed: {previous_cols_dict[col]} → {latest_cols_dict[col]}")

    for col in latest_cols_dict:
        if col not in previous_cols_dict:
            changes.append(f"[ADDED] Column '{col}' was added.")

    prev_order = [col['Name'] for col in previous_cols_list]
    latest_order = [col['Name'] for col in latest_cols_list]

    if prev_order != latest_order:
        changes.append("[ORDER] Column position changed.")
        changes.append(f"Before: {prev_order}")
        changes.append(f"After : {latest_order}")

    if changes:
        full_msg = "\n".join(changes)
        logger.warning("Schema change detected:\n" + full_msg)

        sns.publish(
            TopicArn=sns_topic_arn,
            Subject=f"[Glue Schema Alert] {database_name}.{table_name}",
            Message=full_msg
        )
        if mode == 'FAIL':
            raise Exception("Schema change detected. Job failed intentionally.")
        elif mode == 'AUTO':
            logger.info("Proceeding with auto-accepting schema change.")
        else:
            raise ValueError("Invalid MODE. Use FAIL or AUTO.")
    else:
        logger.info("No schema changes detected.")
else:
    logger.info("No previous versions found. No schema changes to validate.")

job.commit()