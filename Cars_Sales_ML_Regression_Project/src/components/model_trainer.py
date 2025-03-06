import sys
# for regression
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# for classification
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


from src.extra.config_entity import MTConfig
from src.extra.artifact_entity import DTArtifact, MTArtifact
from logger import logging
from exceptions import ProjectException
from src.utils.ml_utils import evaluate_models_regression, evaluate_models_classification
from src.utils.main_utils import save_as_pickle, write_yaml, read_np_array

class ModelTrainer:
    def __init__(self, model_config: MTConfig, data_transformation_artifact: DTArtifact):
        self.model_config = model_config
        self.data_transformation_artifact = data_transformation_artifact

    def set_models_reg(self):
        try:
            models = {
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "DecisionTree": DecisionTreeRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(allow_writing_files=False, verbose=False)
            }

            # params = {
            #     "RandomForest": {
            #         "n_estimators": [100, 300, 500], 
            #         "max_depth": [10, 20, 30], 
            #         "min_samples_split": [2, 5, 10], 
            #         "min_samples_leaf": [1, 2, 5]
            #     },
            #     "GradientBoosting": {
            #         "n_estimators": [100, 300, 500],
            #         "learning_rate": [0.01, 0.1, 0.2], 
            #         "subsample": [0.7, 1.0], 
            #         "max_depth": [3, 7, 9],
            #         "max_features": ["sqrt", "log2"],
            #         "loss": ["ls", "huber"]
            #     },
            #     "AdaBoost": {
            #         "n_estimators": [50, 100, 200],  
            #         "learning_rate": [0.01, 0.1, 1.0]
            #     },
            #     "LinearRegression": {},
            #     "Ridge": {
            #         "alpha": [0.1, 1.0, 10.0]
            #     },
            #     "ElasticNet": {
            #         "alpha": [0.1, 1.0, 10.0],
            #         "l1_ratio": [0.2, 0.5, 0.8]
            #     },
            #     "DecisionTree": {
            #         "max_depth": [3, 5, 10, 20],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 2, 5],
            #         "max_features": ["sqrt", "log2"]
            #     },
            #     "KNeighbors": {
            #         "n_neighbors": [3, 5, 7, 10], 
            #         "weights": ["uniform", "distance"]
            #     },
            #     "SVR": {
            #         "kernel": ["linear", "rbf"], 
            #         "C": [0.1, 1, 10], 
            #         "gamma": ["scale", "auto"]
            #     },
            #     "XGBoost": {
            #         "n_estimators": [100, 300, 500], 
            #         "learning_rate": [0.01, 0.1, 0.2], 
            #         "subsample": [0.7, 1.0], 
            #         "max_depth": [3, 7, 9]
            #     },
            #     "CatBoost": {
            #         "iterations": [100, 300, 500],
            #         "learning_rate": [0.01, 0.1, 0.2], 
            #         "depth": [3, 7, 9]
            #     }
            # }

            params = {
                "RandomForest": {
                    "n_estimators": [100, 200],  
                    "max_depth": [10, 20]
                },
                "GradientBoosting": {
                    "n_estimators": [100, 200],  
                    "learning_rate": [0.05, 0.1]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],  
                    "learning_rate": [0.1, 0.5]
                },
                "LinearRegression": {},
                "Ridge": {
                    "alpha": [0.1, 1.0]
                },
                "ElasticNet": {
                    "alpha": [0.1, 1.0],
                    "l1_ratio": [0.5]
                },
                "DecisionTree": {
                    "max_depth": [5, 10]
                },
                "KNeighbors": {
                    "n_neighbors": [3, 5, 7]
                },
                "SVR": {
                    "kernel": ["rbf"],
                    "C": [1, 10]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],  
                    "max_depth": [3, 6]
                },
                "CatBoost": {
                    "iterations": [100, 200],  
                    "depth": [4, 6]
                }
            }

            return models, params
        except Exception as e:
            logging.error("Error in set_models function {}".format(e))
            raise ProjectException(e, sys)
        
    def train_model_reg(self, X_train, y_train, X_test, y_test):
        try:
            models, params = self.set_models_reg()

            report = evaluate_models_regression(models
                            , params
                            , X_train=X_train
                            , y_train=y_train
                            , X_test=X_test
                            , y_test=y_test)
            
            logging.info("Regression Model training completed")

            best_model_name, best_model_scores = max(report.items(), key=lambda x: x[1]["test_r2_score"])
            logging.info(f"Best regression model scores: {best_model_scores}")
            logging.info(f"Best regression model name: {best_model_name}")

            best_model = models[best_model_name].__class__(**best_model_scores["best_params"])
            logging.info(f"applying best params: {best_model_scores['best_params']} to best regression model: {best_model_name}")
            logging.info(f"fitting X_train, y_train to best regression model: {best_model_name}")
            best_model.fit(X_train, y_train)

            return best_model, report
        
        except Exception as e:
            logging.error("Error in train_model function {}".format(e))
            raise ProjectException(e, sys)
        
    def initiate_model_training_reg(self) -> MTArtifact:
        try:
            logging.info("Starting Model Training")
            train_np_file = self.data_transformation_artifact.transformed_train_file_path
            test_np_file = self.data_transformation_artifact.transformed_test_file_path

            loaded_train_np_file = read_np_array(train_np_file)
            loaded_test_np_file = read_np_array(test_np_file)

            X_train, y_train, X_test, y_test = (
                loaded_train_np_file[:, :-1], 
                loaded_train_np_file[:, -1], 
                loaded_test_np_file[:, :-1], 
                loaded_test_np_file[:, -1]
            )

            logging.info("Regression Model Training started")
            model, report = self.train_model_reg(X_train, y_train, X_test, y_test)

            write_yaml(report, self.model_config.model_report_path)

            logging.info(f"Regression GridSearchCV Models report saved at {self.model_config.model_report_path}")

            save_as_pickle(model, self.model_config.model_path)

            logging.info(f"Regression Model saved at {self.model_config.model_path}")

            logging.info("Regression Model Training completed successfully")

            return MTArtifact(
                model_path=self.model_config.model_path,
                expected_accuracy=self.model_config.expected_accuracy,
                fitting_threshold=self.model_config.fitting_threshold
            )

        except Exception as e:
            logging.error("Error in initiate_model_training function [{}]".format(e))
            raise ProjectException(e, sys)
        

    def set_models_clf(self):
        try:
            models = {
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "LogisticRegression": LogisticRegression(),
                "KNeighbors": KNeighborsClassifier(),
                "SVC": SVC(),
                "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(allow_writing_files=False, verbose=False),
                "GaussianNB": GaussianNB(),
                "AdaBoost": AdaBoostClassifier(),
                "GradientBoosting": GradientBoostingClassifier()
            }

            # params = {
            #     "DecisionTree": {
            #         "criterion": ["gini", "entropy", "log_loss"],
            #         "max_depth": [5, 10, 15, 20],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 2, 5]
            #     },
            #     "RandomForest": {
            #         "n_estimators": [100, 200, 300],
            #         "criterion": ["gini", "entropy"],
            #         "max_depth": [5, 10, 15],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 2, 5]
            #     },
            #     "LogisticRegression": {
            #         "penalty": ["l1", "l2", "elasticnet", "none"],
            #         "C": [0.001, 0.01, 0.1, 1, 10, 100],
            #         "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            #     },
            #     "KNeighbors": {
            #         "n_neighbors": [3, 5, 7, 9, 11],
            #         "weights": ["uniform", "distance"],
            #         "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            #     },
            #     "SVC": {
            #         "C": [0.1, 1, 10, 100],
            #         "kernel": ["linear", "poly", "rbf", "sigmoid"],
            #         "gamma": ["scale", "auto"]
            #     },
            #     "XGBoost": {
            #         "max_depth": [3, 6, 9],
            #         "n_estimators": [100, 200, 300],
            #         "learning_rate": [0.01, 0.1, 0.2, 0.3],
            #         "subsample": [0.5, 0.7, 1.0],
            #         "colsample_bytree": [0.5, 0.7, 1.0],
            #         "reg_alpha": [0, 0.1, 0.5, 1],
            #         "reg_lambda": [0, 0.1, 0.5, 1]
            #     },
            #     "CatBoost": {
            #         "iterations": [100, 200, 300],
            #         "learning_rate": [0.01, 0.1, 0.2, 0.3],
            #         "depth": [3, 6, 9],
            #         "l2_leaf_reg": [1, 3, 5],
            #         "border_count": [32, 64, 128],
            #         "random_strength": [1, 2, 3, 5]
            #     },
            #     "GaussianNB": {},
            #     "AdaBoost": {
            #         "n_estimators": [50, 100, 200],
            #         "learning_rate": [0.01, 0.1, 0.5, 1],
            #         "algorithm": ["SAMME", "SAMME.R"]
            #     }
            #    "GradientBoosting": {
            #        "n_estimators": [100, 200, 300],
            #         "learning_rate": [0.01, 0.1, 0.2, 0.3],
            #         "subsample": [0.5, 0.7, 1.0],
            #         "max_depth": [3, 5, 7],
            #         "max_features": ["sqrt", "log2"],
            #         "loss": ["log_loss", "deviance", "exponential"]
            #     }
            # }
            params = {
                "DecisionTree": {
                    "max_depth": [5, 10],
                    "criterion": ["gini"]
                },
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10],
                    "criterion": ["gini"]
                },
                "LogisticRegression": {
                    "C": [0.1, 1, 10],
                    "solver": ["lbfgs"]
                },
                "KNeighbors": {
                    "n_neighbors": [3, 5, 7]
                },
                "SVC": {
                    "C": [1, 10],
                    "kernel": ["rbf"]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6]
                },
                "CatBoost": {
                    "iterations": [100, 200],
                    "depth": [4, 6]
                },
                "GaussianNB": {},
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1, 0.5]
                },
                "GradientBoosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.5]
                }
            }
            return models, params

        except Exception as e:
            logging.error("Error in initiate_model_training function [{}]".format(e))
            raise ProjectException(e, sys)
    
    def train_model_clf(self, X_train, y_train, X_test, y_test):
        try:
            models, params = self.set_models_clf()

            report = evaluate_models_classification(models
                            , params
                            , X_train=X_train
                            , y_train=y_train
                            , X_test=X_test
                            , y_test=y_test)
            
            logging.info("Classification Model training completed")

            best_model_name, best_model_scores = max(report.items(), key=lambda x: x[1]["test_f1_score"])
            logging.info(f"Best classification model scores: {best_model_scores}")
            logging.info(f"Best classification model name: {best_model_name}")

            best_model = models[best_model_name].__class__(**best_model_scores["best_params"])
            logging.info(f"applying best params: {best_model_scores['best_params']} to best classification model: {best_model_name}")
            logging.info(f"fitting X_train, y_train to best classification model: {best_model_name}")
            best_model.fit(X_train, y_train)

            return best_model, report
        
        except Exception as e:
            logging.error("Error in train_model function {}".format(e))
            raise ProjectException(e, sys)
        
    def initiate_model_training_clf(self) -> MTArtifact:
        try:
            logging.info("Starting Model Training")
            train_np_file = self.data_transformation_artifact.transformed_train_file_path
            test_np_file = self.data_transformation_artifact.transformed_test_file_path

            loaded_train_np_file = read_np_array(train_np_file)
            loaded_test_np_file = read_np_array(test_np_file)

            X_train, y_train, X_test, y_test = (
                loaded_train_np_file[:, :-1], 
                loaded_train_np_file[:, -1], 
                loaded_test_np_file[:, :-1], 
                loaded_test_np_file[:, -1]
            )

            logging.info("Classification Model training started")

            model, report = self.train_model_clf(X_train, y_train, X_test, y_test)

            write_yaml(report, self.model_config.model_report_path)

            logging.info(f"Classification GridSearchCV Models report saved at {self.model_config.model_report_path}")

            save_as_pickle(model, self.model_config.model_path)

            logging.info(f"Classification Model saved at {self.model_config.model_path}")

            logging.info("Classification Model Training completed successfully")

            return MTArtifact(
                model_path=self.model_config.model_path,
                expected_accuracy=self.model_config.expected_accuracy,
                fitting_threshold=self.model_config.fitting_threshold
            )

        except Exception as e:
            logging.error("Error in initiate_model_training function [{}]".format(e))
            raise ProjectException(e, sys)