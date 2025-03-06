from dataclasses import dataclass

@dataclass
class DIArtifact:
    feature_store_file_path: str
    train_file_path: str
    test_file_path: str

@dataclass
class DVArtifact:
    drift_report_file_path: str
    validated_test_file_path: str
    validated_train_file_path: str
    invalid_test_file_path: str
    invalid_train_file_path: str
    validation_status: bool

@dataclass
class DTArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_input_object_file_path: str
    transformed_target_object_file_path: str

@dataclass
class MTArtifact:
    model_path: str
    expected_accuracy: float
    fitting_threshold: float