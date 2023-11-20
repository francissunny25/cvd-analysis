import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def get_data_types(data):
        return list(data.dtypes.astype(str).unique())

def type_splitter(data, type_):
    return list(data.select_dtypes(include=[type_]).columns)

def log(data):
    return np.log(data)

def log1p(data):
    return np.log1p(data)