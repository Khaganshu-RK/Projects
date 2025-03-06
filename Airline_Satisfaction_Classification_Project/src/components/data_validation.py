import os, sys
import pandas as pd
from scipy.stats import ks_2samp

from logger import logging
from exceptions import ProjectException
from src.extra.config_entity import DVConfig
from src.extra.artifact_entity import DIArtifact, DVArtifact
import src.constants as const
from src.utils.main_utils import read_yaml, write_yaml, read_csv

class DataValidation:
    def __init__(self, dv_config: DVConfig, di_artifact: DIArtifact):
        try:
            self.dv_config = dv_config
            self.di_artifact = di_artifact
            self.schema = read_yaml(os.path.join(os.getcwd(), const.PROJECT_NAME, const.SCHEMA_DIR, const.SCHEMA_FILE_NAME))
        except Exception as e:
            logging.error("Error in DataValidation init {}".format(e))
            raise ProjectException(e, sys)
        
    def validate_columns(self, df: pd.DataFrame, filename: str) -> bool:
        try:
            logging.info("Validating columns in the dataframe for {}".format(filename))
            if len(df.columns) == len(self.schema['columns']):
                logging.info("Columns in the {} dataframe are valid as per the schema file, column count is {}".format(filename, len(self.schema['columns'])))
                return True
            logging.error("Columns in the {} dataframe are not valid as per the schema file, column count is {}".format(filename, len(self.schema['columns']))) 
            return False
        except Exception as e:
            logging.error("Error in validating columns in the dataframe {}".format(e))
            raise ProjectException(e, sys)
        
    def detect_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        try:
            logging.info("Detecting drift in the dataframe")
            report = {}
            drift_count = 0
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]
                is_same_distribution = ks_2samp(d1.dropna(), d2.dropna())
                if threshold <= is_same_distribution.pvalue:
                    drift_found = False
                else:
                    drift_found = True
                    drift_count += 1
                report.update({col: {"drift_found": drift_found, "p_value": float(is_same_distribution.pvalue)}})

            drift_report_path = self.dv_config.drift_report_file_path

            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
            write_yaml(report, drift_report_path)
            logging.info("Drift report saved at {}".format(drift_report_path))
            if drift_count > 0:
                logging.error("Drift detected in the dataframe {} columns".format(drift_count))
                return True
            logging.info("No drift detected in the dataframe")
            return False           
        except Exception as e:
            logging.error("Error in detecting drift in the dataframe {}".format(e))
            raise ProjectException(e, sys)
        
    def initiate_data_validation(self) -> DVArtifact:
        try:
            logging.info("Starting Data Validation")
            test_file = self.di_artifact.test_file_path
            train_file = self.di_artifact.train_file_path
            test_df = read_csv(test_file)
            train_df = read_csv(train_file)
            logging.info("Dataframes read successfully")
            is_valid_test_columns = self.validate_columns(test_df, "test")
            is_valid_train_columns = self.validate_columns(train_df, "train")
            is_drift = self.detect_drift(current_df=test_df, base_df=train_df)
            logging.info("Drift detection completed")
            if is_valid_test_columns and is_valid_train_columns and not is_drift:
                logging.info("Data Validation completed successfully")
                os.makedirs(os.path.dirname(self.dv_config.validated_test_data_dir), exist_ok=True)
                test_df.to_csv(self.dv_config.validated_test_data_dir, index=False)
                train_df.to_csv(self.dv_config.validated_train_data_dir, index=False)
                dv_artifact = DVArtifact(
                    drift_report_file_path=self.dv_config.drift_report_file_path,validated_test_file_path=self.dv_config.validated_test_data_dir, validated_train_file_path=self.dv_config.validated_train_data_dir, invalid_test_file_path=None, invalid_train_file_path=None,
                    validation_status=True)
                return dv_artifact
            else:
                logging.error("Data Validation failed")
                os.makedirs(os.path.dirname(self.dv_config.invalid_test_data_dir), exist_ok=True)
                test_df.to_csv(self.dv_config.invalid_test_data_dir, index=False)
                train_df.to_csv(self.dv_config.invalid_train_data_dir, index=False)
                dv_artifact = DVArtifact(
                    drift_report_file_path=self.dv_config.drift_report_file_path,validated_test_file_path=None, validated_train_file_path=None, invalid_test_file_path=self.dv_config.invalid_test_data_dir, invalid_train_file_path=self.dv_config.invalid_train_data_dir,
                    validation_status=False)
        except Exception as e:
            logging.error("Error in initiating data validation {}".format(e))
            raise ProjectException(e, sys)
