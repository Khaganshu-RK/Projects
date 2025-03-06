import sys
from sklearn.metrics import r2_score, accuracy_score, f1_score, precision_score, recall_score

from exceptions import ProjectException
from logger import logging
from src.extra.artifact_entity import MTArtifact, DTArtifact
from src.utils.main_utils import read_pickle, read_np_array

class ModelEvaluation:
    def __init__(self, model_artifact: MTArtifact, dt_artifact: DTArtifact):
        try:
            self.model_artifact = model_artifact
            self.dt_artifact = dt_artifact
        except Exception as e:
            logging.error("Error in ModelEvaluator init {}".format(e))
            raise ProjectException(e, sys)
        
    def evaluate_model_reg(self, model: object, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            return r2
        except Exception as e:
            logging.error("Error in evaluate_model function {}".format(e))
            raise ProjectException(e, sys)
        
    def initiate_model_evaluation_reg(self):
        try:
            logging.info("Starting Model Evaluation")
            model_path = self.model_artifact.model_path

            model = read_pickle(model_path)

            test_np_file = self.dt_artifact.transformed_test_file_path

            test_np_loaded = read_np_array(test_np_file)

            X_test, y_test = (
                test_np_loaded[:, :-1], 
                test_np_loaded[:, -1]
            )   

            r2 = self.evaluate_model_reg(model, X_test, y_test)

            if r2 < self.model_artifact.expected_accuracy:
                logging.error("Model accuracy is less than expected accuracy threshold of {}".format(self.model_artifact.expected_accuracy))
                return False

            logging.info(f"Model accuracy is: {r2}")

            logging.info("Model Evaluation completed successfully")
            return True
        
        except Exception as e:
            logging.error("Error in initiate_model_evaluation function [{}]".format(e))
            raise ProjectException(e, sys)
        
    def evalute_model_clf(self, model: object, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            score_f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            return score_f1, accuracy, precision, recall
        except Exception as e:
            logging.error("Error in evaluate_model function {}".format(e))
            raise ProjectException(e, sys)
        
    def initiate_model_evaluation_clf(self):
        try:
            logging.info("Starting Model Evaluation")
            model_path = self.model_artifact.model_path

            model = read_pickle(model_path)

            test_np_file = self.dt_artifact.transformed_test_file_path

            test_np_file_loaded = read_np_array(test_np_file)

            X_test, y_test = (
                test_np_file_loaded[:, :-1], 
                test_np_file_loaded[:, -1]
            )

            score_f1, accuracy, precision, recall = self.evalute_model_clf(model, X_test, y_test)

            if score_f1 < self.model_artifact.expected_accuracy:
                logging.error("Model accuracy is less than expected accuracy threshold of {}".format(self.model_artifact.expected_accuracy))
                return False

            logging.info(f"Model accuracy is: {accuracy}")
            logging.info(f"Model precision is: {precision}")
            logging.info(f"Model recall is: {recall}")
            logging.info(f"Model f1 score is: {score_f1}")

            logging.info("Model Evaluation completed successfully")
            return True
        
        except Exception as e:    
            logging.error("Error in initiate_model_evaluation function [{}]".format(e)) 
            raise ProjectException(e, sys)