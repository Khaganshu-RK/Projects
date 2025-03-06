import sys
import pandas as pd
import numpy as np

from src.pipelines.pipeline import full_pipeline
from src.extra.config_entity import DTConfig
from src.extra.artifact_entity import DVArtifact, DTArtifact
from src.utils.main_utils import read_csv, save_as_pickle, save_as_np_array
from logger import logging
from exceptions import ProjectException
import src.constants as const

class DataTransformation:
    def __init__(self, dt_config: DTConfig, dv_artifact: DVArtifact):
        self.dt_config = dt_config
        self.dv_artifact = dv_artifact
        
    def transform_data(self, train_input_features_df: pd.DataFrame, test_input_features_df: pd.DataFrame, y_train: pd.DataFrame):
        try:
            logging.info("Transforming data")
            logging.info("Encoding the target variable")
            logging.info("Creating the pipeline")
            preprocessor, y_encoder_model, y_train_transformed = full_pipeline(train_input_features_df, y_train, const.PRED_TYPE)
            logging.info("Fitting and transforming the train data")
            transformed_train_input_features_array = preprocessor.fit_transform(train_input_features_df)
            logging.info("Transforming the test data")
            transformed_test_input_features_array = preprocessor.transform(test_input_features_df)
            return y_encoder_model, preprocessor, transformed_train_input_features_array, transformed_test_input_features_array, y_train_transformed
        except Exception as e:
            logging.error("Error in transforming data {}".format(e))
            raise ProjectException(e, sys)
        
    def initiate_data_transformation(self) -> DTArtifact:
        try:
            logging.info("Starting Data Transformation")
            train_df = read_csv(self.dv_artifact.validated_train_file_path)
            test_df = read_csv(self.dv_artifact.validated_test_file_path)

            train_input_features_df = train_df.drop(columns=[const.TARGET_COL], axis=1)
            target_train_df = train_df[const.TARGET_COL]

            test_input_features_df = test_df.drop(columns=[const.TARGET_COL], axis=1)
            target_test_df = test_df[const.TARGET_COL]
            
            y_encoder_model, preprocessor, transformed_train_input_features_array, transformed_test_input_features_array, y_train_transformed = self.transform_data(train_input_features_df, test_input_features_df, target_train_df)

            if y_encoder_model is not None:
                y_test_transformed = y_encoder_model.transform(target_test_df)
                save_as_pickle(y_encoder_model, self.dt_config.transformed_target_object_dir)
                test_array = np.c_[transformed_test_input_features_array, y_test_transformed]
                train_array = np.c_[transformed_train_input_features_array, y_train_transformed]
            else:
                train_array = np.c_[transformed_train_input_features_array, np.array(target_train_df)]
                test_array = np.c_[transformed_test_input_features_array, np.array(target_test_df)]
            

            save_as_np_array(train_array, self.dt_config.transformed_train_data_path)
            save_as_np_array(test_array, self.dt_config.transformed_test_data_path)

            save_as_pickle(preprocessor, self.dt_config.transformed_input_object_dir)

            logging.info("Data Transformation completed successfully")

            dt_artifact = DTArtifact(self.dt_config.transformed_train_data_path, self.dt_config.transformed_test_data_path, self.dt_config.transformed_input_object_dir, self.dt_config.transformed_target_object_dir if y_encoder_model is not None else None)

            return dt_artifact
        except Exception as e:
            logging.error("Error in initiating data transformation [{}]".format(e))
            raise ProjectException(e, sys)
