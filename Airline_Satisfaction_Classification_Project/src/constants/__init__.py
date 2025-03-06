# Description: This file contains all the constants used in the project.
#
"""
Common constants
"""

PROJECT_NAME: str = 'Airline_Satisfaction_Classification_Project'
TARGET_COL: str = 'satisfaction'
PRED_TYPE: str = 'binary_classification' # regression, binary_classification, multi_classification

PIPELINE_NAME: str = 'airline_satisfaction_pipeline'

S3_BUCKET_NAME: str = 'krkprojects'
S3_MODEL_DIR: str = 'model'
S3_REPORT_DIR: str = 'report'

ARTIFACT_DIR: str = 'Artifacts'
FILE_NAME: str = 'cleaned_Data.csv'
SCHEMA_DIR: str = 'Schema'
SCHEMA_FILE_NAME: str = 'data_schema.yaml'

TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'


"""
Data Ingestion (DI) related constant
"""
DI_DIR_NAME: str = '01_Data_Ingestion'
DI_FEATURE_STORE_DIR: str = 'feature_store'
DI_INGESTED_DIR: str = 'ingested'
DI_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation (DV) related constant
"""
DV_DIR_NAME: str = '02_Data_Validation'
DV_DRIFT_REPORT_DIR: str = 'drift_report'
DV_DRIFT_REPORT_FILE_NAME: str = 'drift_report.yaml'
DV_VALIDATED_DIR: str = 'validated'
DV_INVALID_DIR: str = 'invalid'

"""
Data Transformation (DT) related constant
"""
DT_DIR_NAME: str = '03_Data_Transformation'
DT_TRANSFORMED_DATA_DIR: str = 'transformed'
DT_TRAIN_FILE_NAME: str = 'train.npy'
DT_TEST_FILE_NAME: str = 'test.npy'
DT_TRANSFORMED_OBJECT_DIR: str = 'transformed_object'
DT_PREPROCESSOR_FILE_NAME: str = 'input_preprocessor.pkl'
DT_TARGET_ENCODER_FILE_NAME: str = 'target_encoder.pkl'

"""
Model Training (MT) related constant
"""
MT_DIR_NAME: str = '04_Model_Training'
MT_MODEL_DIR: str = 'final_model'
MT_MODEL_FILE_NAME: str = 'model.pkl'
MT_MODEL_REPORT_DIR: str = 'models_report'
MT_MODEL_REPORT_FILE_NAME: str = 'model_report.yaml'
MT_FITTING_THRESHOLD: float = 0.05
MT_TRAINER_SCORE: float = 0.6

