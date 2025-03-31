import os, sys
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from logger import logging
from exceptions import ProjectException
import src.constants as const
from src.extra.config_entity import DIConfig
from src.extra.artifact_entity import DIArtifact
from databases import Databases

class DataIngestion:
    def __init__(self, di_config: DIConfig):
        self.di_config = di_config

    def copy_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Copying file from {const.FILE_NAME} to {self.di_config.feature_store_file_path}")
            source_file_path = os.path.join(os.getcwd(), const.PROJECT_NAME, "Data", const.FILE_NAME)
            feature_store_dir = self.di_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_dir), exist_ok=True)

            shutil.copyfile(source_file_path, feature_store_dir)

            logging.info(f"Reading data from {feature_store_dir}")
            df=pd.read_csv(feature_store_dir)

            return df
        except Exception as e:
            logging.error(f"Error in copying file from {const.FILE_NAME} to {feature_store_dir}")
            raise ProjectException(e, sys)
    
    def test_train_split(self, data: pd.DataFrame):
        try:
            train, test = train_test_split(data, test_size=self.di_config.train_test_split_ratio, random_state=42)

            dir_path = os.path.dirname(self.di_config.ingested_train_data_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Splitting data into train and test and saving to {self.di_config.ingested_train_data_path} and {self.di_config.ingested_test_data_path}")

            train.to_csv(self.di_config.ingested_train_data_path, index=False, header=True)

            test.to_csv(self.di_config.ingested_test_data_path, index=False, header=True)
        except Exception as e:
            logging.error(f"Error in splitting data into train and test")
            raise ProjectException(e, sys)
        
    def ingest_data(self): 
        try:
            logging.info("Starting Data Ingestion") 
            # data = self.copy_data()
            os.makedirs(os.path.dirname(self.di_config.feature_store_file_path), exist_ok=True)
            logging.info("Reading data from MongoDB and PostgreSQL")
            data_mongodb = Databases("mongodb")
            data_m = data_mongodb.read_data("mongodb")
            data_m = pd.DataFrame(list(data_m))
            print(data_m.head())

            logging.info("Reading data from PostgreSQL")
            data_postgresql = Databases("postgresql")
            data_p, column_names = data_postgresql.read_data("postgresql")
            data_p = pd.DataFrame(data_p, columns=column_names)
            print(data_p.head())

            logging.info("Concatenating data from MongoDB and PostgreSQL")
            # joining the data from mongodb and postgresql vertically
            data = pd.concat([data_m, data_p], axis=0, ignore_index=True)
            data.to_csv(self.di_config.feature_store_file_path, index=False, header=True)
            print(data.head())
            self.test_train_split(data)
            logging.info("Data Ingestion completed successfully")
            return DIArtifact(self.di_config.feature_store_file_path, self.di_config.ingested_train_data_path, self.di_config.ingested_test_data_path)
        except Exception as e:
            logging.error("Error in Ingestion [{}]".format(e))
            raise ProjectException(e, sys)