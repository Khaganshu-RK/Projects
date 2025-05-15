# AWS Event-Driven Data Engineering Pipeline

This repository contains an end-to-end, event-driven data engineering pipeline on AWS that handles data ingestion from on-premise systems, schema validation, transformation using Delta Lake, aggregation using EMR, and querying via Redshift Spectrum. All tasks are orchestrated using Apache Airflow on MWAA (Managed Workflows for Apache Airflow).

## Key Components

- Automated S3 ingestion using Lambda or AWS DataSync
- Schema validation using AWS Glue Jobs
- Delta Lake transformations using PySpark (Glue/EMR)
- Aggregation and analytics using EMR and Redshift Spectrum
- DAG orchestration via Apache Airflow (MWAA)
- Alerting via Amazon SNS and CloudWatch logging

## Architecture Overview

1. Files are ingested from on-premise servers using AWS DataSync or custom Lambda.
2. Files are partitioned by date and stored in S3.
3. A Lambda function is triggered via S3 PUT event (via EventBridge).
4. This Lambda function triggers an Airflow DAG.
5. The DAG executes:
   - A Glue job to validate schema changes.
   - A Glue job to transform data into Delta format.
   - An EMR job to perform aggregations.
   - A Redshift Spectrum query to analyze results.

## Technologies Used

- Amazon S3
- AWS Lambda
- AWS DataSync
- Amazon EventBridge
- AWS Glue (Crawlers, Jobs, Catalog)
- AWS EMR (PySpark)
- Amazon Redshift (Spectrum)
- Apache Airflow (MWAA)
- Amazon SNS
- Amazon CloudWatch

## Pipeline Steps

1. **Ingestion**

   - Files are uploaded to S3 from an on-premise server.
   - Lambda partitions the data by year/month/day.
   - EventBridge triggers the MWAA DAG.

2. **Schema Validation**

   - A Glue Job compares the latest crawled schema to the previous version.
   - If schema changes are detected (name, type, or position), the job fails and triggers an SNS notification.

3. **Data Transformation**

   - Validated data is transformed and written in Delta format using a Glue or EMR job.
   - Transformed data is saved to a processed S3 location.

4. **Aggregation**

   - An EMR job reads Delta files and performs aggregations such as:
     - Count of products ordered by each user.
     - Top 20 customers with the highest number of orders.
   - Aggregated data is saved back to S3 in Delta format.

5. **Analytics**

   - Redshift Spectrum reads the final Delta files using external tables.
   - Analytical queries can be written directly in Redshift.

6. **Orchestration**

   - MWAA DAG controls the workflow using Python code.
   - Each step is implemented as an Airflow task with retry policies and failure alerts.

7. **Alerting**
   - SNS notifications for alerting.
   - CloudWatch logs are used for monitoring.

## Features

- Event-driven, serverless architecture
- Schema drift detection with automatic fail and notify
- Delta Lake support for optimized S3 querying
- Scalable transformation and aggregation using Spark
- BI-ready output with Redshift Spectrum
- Logging and observability using CloudWatch
- Notifications via SNS

## Monitoring and Alerts

- CloudWatch Logs capture Lambda, Glue, and Airflow output
- SNS topic triggers email alert if schema drift is detected
- All job statuses are visible in Airflow UI

## Security and IAM

- Lambda: Access to S3, EventBridge, and MWAA
- Glue: Access to S3, Glue Catalog, SNS
- EMR: Roles for reading/writing S3 and running Spark jobs
- Redshift: External schema permissions for S3 access

## Deployment Notes

- Ensure all IAM roles are configured correctly
- Set up Glue Crawlers for the raw S3 paths
- Redshift external schemas must point to transformed S3 locations
- MWAA environment must have access to S3, Glue, EMR, Redshift, and SNS

## Future Enhancements

- Add data quality checks with Great Expectations
- Implement schema registry with AWS Glue Schema Registry
- CI/CD integration for code deployment and job versioning
- Use Step Functions for more complex orchestration if needed

## License

This project is licensed under the MIT License.
