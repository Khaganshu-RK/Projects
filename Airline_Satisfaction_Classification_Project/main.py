import sys, os

from logger import logging
from exceptions import ProjectException
from src.extra.config_entity import CommonConfig, DIConfig, DVConfig, DTConfig, MTConfig
from src.components import data_ingestion, data_validation, data_transformation, model_trainer, model_evaluation
from src.utils.cloud_utils import s3_upload, s3_download
import src.constants as const

if __name__ == "__main__":
    try:
        logging.info("Main started")
        common_config = CommonConfig()

        di_config = DIConfig(common_config)
        di = data_ingestion.DataIngestion(di_config)
        di_artifact = di.ingest_data()

        dv_config = DVConfig(common_config)
        dv = data_validation.DataValidation(dv_config, di_artifact)
        dv_artifact = dv.initiate_data_validation()

        if dv_artifact.validation_status:

            dt_config = DTConfig(common_config)
            dt = data_transformation.DataTransformation(dt_config, dv_artifact)
            dt_artifact = dt.initiate_data_transformation()

            mt_config = MTConfig(common_config)
            mt = model_trainer.ModelTrainer(mt_config, dt_artifact)
            mt_artifact = mt.initiate_model_training_clf()

            me = model_evaluation.ModelEvaluation(model_artifact=mt_artifact, dt_artifact=dt_artifact)
            me.initiate_model_evaluation_clf()

            logging.info("AWS Upload Started")

            logging.info("Uploading model from local to s3 bucket")
            s3_upload(bucket_name=const.S3_BUCKET_NAME, file_path=mt_artifact.model_path, object_name=os.path.join(const.PROJECT_NAME,const.S3_MODEL_DIR, const.MT_MODEL_FILE_NAME))
            logging.info("Uploading drift report from local to s3 bucket")
            s3_upload(bucket_name=const.S3_BUCKET_NAME, file_path=dv_artifact.drift_report_file_path, object_name=os.path.join(const.PROJECT_NAME,const.S3_REPORT_DIR, const.DV_DRIFT_REPORT_FILE_NAME))

            logging.info("Uploading preprocessor from local to s3 bucket")
            s3_upload(bucket_name=const.S3_BUCKET_NAME, file_path=dt_artifact.transformed_input_object_file_path, object_name=os.path.join(const.PROJECT_NAME,const.S3_MODEL_DIR, const.DT_PREPROCESSOR_FILE_NAME))

            logging.info("Uploading GridSearchCV models report from local to s3 bucket")
            s3_upload(bucket_name=const.S3_BUCKET_NAME, file_path=mt_config.model_report_path, object_name=os.path.join(const.PROJECT_NAME,const.S3_REPORT_DIR, const.MT_MODEL_REPORT_FILE_NAME))
            logging.info("Uploading data schema from local to s3 bucket")
            s3_upload(bucket_name=const.S3_BUCKET_NAME, file_path=os.path.join(os.getcwd(), const.PROJECT_NAME, const.SCHEMA_DIR, const.SCHEMA_FILE_NAME), object_name=os.path.join(const.PROJECT_NAME,const.SCHEMA_DIR, const.SCHEMA_FILE_NAME))

            if dt_artifact.transformed_target_object_file_path is not None:
                logging.info("Uploading target encoder from local to s3 bucket")
                s3_upload(bucket_name=const.S3_BUCKET_NAME, file_path=dt_artifact.transformed_target_object_file_path, object_name=os.path.join(const.PROJECT_NAME,const.S3_MODEL_DIR, const.DT_TARGET_ENCODER_FILE_NAME))

            logging.info("AWS Upload Completed")

            logging.info("AWS Download Started")

            logging.info("Downloading best model from s3 bucket to local for backend")
            s3_download(bucket_name=const.S3_BUCKET_NAME, object_name=os.path.join(const.PROJECT_NAME,const.S3_MODEL_DIR, const.MT_MODEL_FILE_NAME), file_path=os.path.join(os.getcwd(), const.PROJECT_NAME, "Web", "backend", "app", "data" , const.MT_MODEL_FILE_NAME))

            logging.info("Downloading preprocessor from s3 bucket to local for backend")
            s3_download(bucket_name=const.S3_BUCKET_NAME, object_name=os.path.join(const.PROJECT_NAME,const.S3_MODEL_DIR, const.DT_PREPROCESSOR_FILE_NAME), file_path=os.path.join(os.getcwd(), const.PROJECT_NAME, "Web", "backend", "app", "data" , const.DT_PREPROCESSOR_FILE_NAME))

            logging.info("Downloading data schema from s3 bucket for backend")
            s3_download(bucket_name=const.S3_BUCKET_NAME, object_name=os.path.join(const.PROJECT_NAME,const.SCHEMA_DIR, const.SCHEMA_FILE_NAME), file_path=os.path.join(os.getcwd(), const.PROJECT_NAME, "Web", "backend", "app", "data" , const.SCHEMA_FILE_NAME))

            logging.info("Downloading data schema from s3 bucket for frontend")
            s3_download(bucket_name=const.S3_BUCKET_NAME, object_name=os.path.join(const.PROJECT_NAME,const.SCHEMA_DIR, const.SCHEMA_FILE_NAME), file_path=os.path.join(os.getcwd(), const.PROJECT_NAME, "Web", "frontend", "app", "data", const.SCHEMA_FILE_NAME))
            
            if dt_artifact.transformed_target_object_file_path is not None:
                logging.info("Downloading target encoder from s3 bucket to local for backend")
                s3_download(bucket_name=const.S3_BUCKET_NAME, object_name=os.path.join(const.PROJECT_NAME,const.S3_MODEL_DIR, const.DT_TARGET_ENCODER_FILE_NAME), file_path=os.path.join(os.getcwd(), const.PROJECT_NAME, "Web", "backend", "app", "data" , const.DT_TARGET_ENCODER_FILE_NAME))

            logging.info("AWS Download Completed")

            logging.info("Main Ended Successfully")
        else:
            raise Exception("Data validation failed")

    except Exception as e:
        logging.error("Error in main function [{}]".format(e))
        raise ProjectException(e, sys)
