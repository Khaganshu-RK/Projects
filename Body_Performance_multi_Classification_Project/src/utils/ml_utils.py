from logger import logging
import sys
from sklearn.metrics import r2_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from exceptions import ProjectException


def evaluate_models_regression(models, params, X_train, y_train, X_test, y_test):
    """
    Evaluate multiple models using GridSearchCV and return their performance.
    
    Args:
        models (dict): Dictionary of model names and model objects.
        params (dict): Dictionary of model names and hyperparameter grids.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
    
    Returns:
        dict: Model names with their train and test R2 scores.
    """
    report = {}
    try:
        for model_name, model in models.items():
            model_params = params.get(model_name, {})

            logging.info(f"Evaluating regression model: {model_name} - {model}")
            # Grid search for hyperparameter tuning
            gs_clf = GridSearchCV(model, model_params, n_jobs=-1, cv=3, verbose=0)
            gs_clf.fit(X_train, y_train)

            # Set the best parameters and refit the model
            model.set_params(**gs_clf.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            r2_score_train = r2_score(y_train, y_train_pred)
            r2_score_test = r2_score(y_test, y_test_pred)

            # Update report
            report[model_name] = {"train_r2_score": r2_score_train, "test_r2_score": r2_score_test, "best_params": gs_clf.best_params_}
            logging.info(f"Model: {model_name}, Train R2 score: {r2_score_train}, Test R2 score: {r2_score_test}")

        return report

    except Exception as e:
        logging.error(f"Error in evaluating models: {e}")
        raise ProjectException(e, sys)
    

def evaluate_models_classification(models, params, X_train, y_train, X_test, y_test):
    """
    Evaluate multiple models using GridSearchCV and return their performance.
    
    Args:
        models (dict): Dictionary of model names and model objects.
        params (dict): Dictionary of model names and hyperparameter grids.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
    
    Returns:
        dict: Model names with their train and test accuracy scores.
    """
    report = {}
    try:
        for model_name, model in models.items():
            model_params = params.get(model_name, {})

            logging.info(f"Evaluating classification model: {model_name} - {model}")
            # Grid search for hyperparameter tuning
            gs_clf = GridSearchCV(model, model_params, n_jobs=-1, cv=3, verbose=0)
            gs_clf.fit(X_train, y_train)

            # Set the best parameters and refit the model
            model.set_params(**gs_clf.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calulate the cofusion matrix and accuracy scores and f1 scores and precision scores and recall scores
            # r2_score_train = r2_score(y_train, y_train_pred, multioutput='variance_weighted')
            # r2_score_test = r2_score(y_test, y_test_pred, multioutput='variance_weighted')
            accuracy_test = accuracy_score(y_test, y_test_pred)
            cm_test = confusion_matrix(y_test, y_test_pred)
            f1_test = f1_score(y_test, y_test_pred, average='weighted')
            precision_test = precision_score(y_test, y_test_pred, average='weighted')
            recall_test = recall_score(y_test, y_test_pred, average='weighted')

            # Update report
            report[model_name] = {
                # "train_r2_score": r2_score_train,
                # "test_r2_score": r2_score_test,
                "test_accuracy": accuracy_test,
                "test_confusion_matrix": cm_test,
                "test_f1_score": f1_test,
                "test_precision_score": precision_test,
                "test_recall_score": recall_test,
                "best_params": gs_clf.best_params_
            }

            # logging.info(f"Model: {model_name}, Train R2 score: {r2_score_train}, Test R2 score: {r2_score_test}, Test accuracy: {accuracy_test}, Test f1 score: {f1_test}, Test precision score: {precision_test}, Test recall score: {recall_test}")

            logging.info(f"Model: {model_name}, Test accuracy: {accuracy_test}, Test confusion matrix: {cm_test}, Test f1 score: {f1_test}, Test precision score: {precision_test}, Test recall score: {recall_test}")

        return report

    except Exception as e:
        logging.error(f"Error in evaluating models: {e}")
        raise ProjectException(e, sys)