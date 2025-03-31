import pickle, yaml, os, sys
import pandas as pd
import numpy as np

from exceptions import ProjectException
from logger import logging

def read_yaml(file_path: str) -> dict:
    try:
        logging.info("Reading yaml file from {}".format(file_path))
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error("Error in reading yaml file {}".format(e))
        raise ProjectException(e, sys)
    
def write_yaml(data: object, file_path: str) -> None:
    try:
        logging.info("Writing yaml file to {}".format(file_path))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
    except Exception as e:
        logging.error("Error in writing yaml file {}".format(e))
        raise ProjectException(e, sys)

def read_csv(file_path: str) -> pd.DataFrame:
    try:
        logging.info("Reading csv file from {}".format(file_path))
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error("Error in reading csv file {}".format(e))
        raise ProjectException(e, sys)
    
def save_as_np_array(data: np.ndarray, file_path: str) -> None:
    try:
        logging.info("Saving data as numpy array at {}".format(file_path))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, data)
    except Exception as e:
        logging.error("Error in saving data as numpy array {}".format(e))
        raise ProjectException(e, sys)
    
def read_np_array(file_path: str) -> np.ndarray:
    try:
        logging.info("Reading numpy array from {}".format(file_path))
        with open(file_path, 'rb') as file:
            return np.load(file, allow_pickle=True)
    except Exception as e:
        logging.error("Error in reading numpy array {}".format(e))
        raise ProjectException(e, sys)
    
def save_as_pickle(data: object, file_path: str) -> None:
    try:
        logging.info("Saving data as pickle at {}".format(file_path))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        logging.error("Error in saving data as pickle {}".format(e))
        raise ProjectException(e, sys)
    
def read_pickle(file_path: str) -> object:
    try:
        logging.info("Reading pickle file from {}".format(file_path))
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        logging.error("Error in reading pickle file {}".format(e))
        raise ProjectException(e, sys)
    