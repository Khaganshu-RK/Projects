import os
from datetime import datetime
import src.constants as const

"""
Common configuration for the project containg timestamp, artifact directory and model directory for the project.

Why we are storing the paths in the configuration file?
- To make sure that the paths are consistent across the project.
- To make sure that the paths are not hardcoded in the code.
"""
class CommonConfig:
    def __init__(self):
        self.timestamp: str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        self.artifact_dir: str = os.path.join(os.getcwd(), const.PROJECT_NAME, const.ARTIFACT_DIR, self.timestamp)
"""
Data Ingestion configuration for the project containing feature store file path, ingested train data path, ingested test data path and train test split ratio.
"""
class DIConfig:
    def __init__(self, common_config: CommonConfig):
        self.data_ingestion_dir: str = os.path.join(common_config.artifact_dir, const.DI_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, const.DI_FEATURE_STORE_DIR, const.FILE_NAME)
        self.ingested_train_data_path: str = os.path.join(self.data_ingestion_dir, const.DI_INGESTED_DIR, const.TRAIN_FILE_NAME)
        self.ingested_test_data_path: str = os.path.join(self.data_ingestion_dir, const.DI_INGESTED_DIR, const.TEST_FILE_NAME)
        self.train_test_split_ratio: float = const.DI_TRAIN_TEST_SPLIT_RATIO

"""
Data Validation configuration for the project containing drift report file path, validated test data path, validated train data path, invalid test data path, invalid train data path file path.
"""
class DVConfig:
    def __init__(self, common_config: CommonConfig):
        self.data_validation_dir: str = os.path.join(common_config.artifact_dir, const.DV_DIR_NAME)
        self.drift_report_file_path: str = os.path.join(self.data_validation_dir, const.DV_DRIFT_REPORT_DIR, const.DV_DRIFT_REPORT_FILE_NAME)
        self.validated_test_data_dir: str = os.path.join(self.data_validation_dir, const.DV_VALIDATED_DIR, const.TEST_FILE_NAME)
        self.validated_train_data_dir: str = os.path.join(self.data_validation_dir, const.DV_VALIDATED_DIR, const.TRAIN_FILE_NAME)
        self.invalid_test_data_dir: str = os.path.join(self.data_validation_dir, const.DV_INVALID_DIR, const.TEST_FILE_NAME)
        self.invalid_train_data_dir: str = os.path.join(self.data_validation_dir, const.DV_INVALID_DIR, const.TRAIN_FILE_NAME)

"""
Data Transformation configuration for the project containing transformed train data path, transformed test data path and transformed object directory.
"""
class DTConfig:
    def __init__(self, common_config: CommonConfig):
        self.data_transformation_dir: str = os.path.join(common_config.artifact_dir, const.DT_DIR_NAME)
        self.transformed_train_data_path: str = os.path.join(self.data_transformation_dir, const.DT_TRANSFORMED_DATA_DIR, const.DT_TRAIN_FILE_NAME)
        self.transformed_test_data_path: str = os.path.join(self.data_transformation_dir, const.DT_TRANSFORMED_DATA_DIR, const.DT_TEST_FILE_NAME)
        self.transformed_input_object_dir: str = os.path.join(self.data_transformation_dir, const.DT_TRANSFORMED_OBJECT_DIR, const.DT_PREPROCESSOR_FILE_NAME)
        self.transformed_target_object_dir: str = os.path.join(self.data_transformation_dir, const.DT_TRANSFORMED_OBJECT_DIR, const.DT_TARGET_ENCODER_FILE_NAME)

"""
Model Training configuration for the project containing model training directory, model path, expected accuracy and fitting threshold.
"""
class MTConfig:
    def __init__(self, common_config: CommonConfig):
        self.model_training_dir: str = os.path.join(common_config.artifact_dir, const.MT_DIR_NAME)
        self.model_path: str = os.path.join(self.model_training_dir, const.MT_MODEL_DIR, const.MT_MODEL_FILE_NAME)
        self.model_report_path: str = os.path.join(self.model_training_dir, const.MT_MODEL_REPORT_DIR, const.MT_MODEL_REPORT_FILE_NAME)
        self.expected_accuracy: float = const.MT_TRAINER_SCORE
        self.fitting_threshold: float = const.MT_FITTING_THRESHOLD
