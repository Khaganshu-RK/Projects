from airflow import DAG
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.operators.lambda_function import LambdaInvokeFunctionOperator
from airflow.providers.amazon.aws.operators.emr_add_steps import EmrAddStepsOperator
from airflow.providers.amazon.aws.operators.emr import EmrStepSensor
from airflow.providers.amazon.aws.operators.redshift_data import RedshiftDataOperator
from airflow.providers.amazon.aws.operators.sns import SnsPublishOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import timedelta

# Default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['data-team@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id='etl_pipeline_mwaa',
    default_args=default_args,
    description='End-to-end ETL pipeline orchestrated by MWAA',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['etl', 'mwaa', 'aws'],
) as dag:

    validate_schema = LambdaInvokeFunctionOperator(
        task_id='validate_schema',
        function_name='validate_schema_lambda',
        payload={'TABLE_NAME': 'table_name', 'DATABASE_NAME': 'db_name' , 'SNS_TOPIC_ARN': 'sns_topic_arn', 'MODE': 'FAIL'}, 
        log_type='Tail',
    )

    run_schema_validation = GlueJobOperator(
        task_id='run_schema_validation',
        job_name='schema_validation_job',
        script_location='s3://path-to-your-script/schema_validation.py',
        region_name='your-region',
        iam_role_name='your-glue-role',
        num_of_dpus=10,
    )

    run_data_transformation = GlueJobOperator(
        task_id='run_data_transformation',
        job_name='data_transformation_job',
        script_location='s3://path-to-your-script/data_transformation.py',
        region_name='your-region',
        iam_role_name='your-glue-role',
        num_of_dpus=10,
    )

    emr_steps = [
        {
            'Name': 'Process Delta Data',
            'ActionOnFailure': 'CONTINUE',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': [
                    'spark-submit',
                    '--deploy-mode', 'cluster',
                    '--class', 'org.apache.spark.examples.SparkPi',
                    's3://path-to-your-script/emr_processing.py',
                    '--arg1', 'value1',
                    '--arg2', 'value2',
                ],
            },
        },
    ]

    add_emr_steps = EmrAddStepsOperator(
        task_id='add_emr_steps',
        job_flow_id='emr-cluster-id',
        steps=emr_steps,
    )

    monitor_emr_step = EmrStepSensor(
        task_id='monitor_emr_step',
        job_flow_id='emr-cluster-id',
        step_id="{{ task_instance.xcom_pull(task_ids='add_emr_steps', key='return_value')[0] }}",
    )

    load_into_redshift = RedshiftDataOperator(
        task_id='load_into_redshift',
        cluster_identifier='redshift-cluster-id',
        database='your_database',
        db_user='your_db_user',
        sql='sql/load_data.sql',
    )

    notify_completion = SnsPublishOperator(
        task_id='notify_completion',
        target_arn='arn:aws:sns:region:account-id:topic-name',
        message='ETL pipeline completed successfully.',
        subject='ETL Pipeline Notification',
    )

    validate_schema >> run_schema_validation >> run_data_transformation
    run_data_transformation >> add_emr_steps >> monitor_emr_step
    monitor_emr_step >> load_into_redshift >> notify_completion