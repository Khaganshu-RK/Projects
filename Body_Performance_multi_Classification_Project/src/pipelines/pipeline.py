from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def preprocessing_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:
    numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()) 
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough')

    return preprocessor


# def feature_selection_pipeline(X_train: pd.DataFrame, y_train: pd.DataFrame, pred_type: str) -> Pipeline:
#     """Applies feature selection based on prediction type."""
#     if pred_type == 'regression':
#         feature_selection_pipeline = Pipeline(steps=[
#             ('feature_selector', SelectFromModel(Lasso(alpha=0.01, random_state=0)))
#         ])
#         return feature_selection_pipeline
    
#     elif pred_type == 'binary_classification':
#         feature_selection_pipeline = Pipeline(steps=[
#             ('feature_selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1)))
#         ])
#         return feature_selection_pipeline
    
#     elif pred_type == 'multi_classification':
#         feature_selection_pipeline = Pipeline(steps=[
#             ('feature_selector', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=0)))
#         ])
#         return feature_selection_pipeline
    
#     else:
#         raise ValueError("Invalid prediction type. Choose from 'regression', 'binary_classification', 'multi_classification'.")

# def pca_pipeline(X_train_transformed: pd.DataFrame) -> Pipeline:
#     """Applies PCA for dimensionality reduction."""
#     pca_pipeline = Pipeline(steps=[
#         ('pca', PCA(n_components=0.95))
#     ])
#     return pca_pipeline


def full_pipeline(X_train: pd.DataFrame, y_train: pd.DataFrame, pred_type: str) -> tuple:
    """Creates the full pipeline with preprocessing, feature selection, and PCA."""
    
    # Encode target variable
    label_encoder, y_train_transformed = y_value_encoder(y_train, pred_type)

    # Define pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing_pipeline(X_train)),  
    ])

    return pipeline, label_encoder, y_train_transformed


def y_value_encoder(y_train: pd.DataFrame, pred_type: str):
    """Encodes target variable based on prediction type."""
    if pred_type == 'regression':
        return None, y_train.values.ravel()  # Ensure it's 1D
    
    elif pred_type == 'binary_classification':
        label_encoder = LabelEncoder()
        if not np.issubdtype(y_train.dtype, np.integer):  # Check if dtype is not int
            y_train = y_train.astype(str)
        y_train_transformed = label_encoder.fit_transform(y_train.values.ravel())  # Flatten to 1D
        return label_encoder, y_train_transformed

    elif pred_type == 'multi_classification':
        ohe_encoder = OneHotEncoder(sparse_output=False)
        y_train_transformed = ohe_encoder.fit_transform(y_train.values.reshape(-1, 1))  # Reshape to 2D
        return ohe_encoder, y_train_transformed
    
    else:
        raise ValueError("Invalid prediction type. Choose from 'regression', 'binary_classification', 'multi_classification'.")