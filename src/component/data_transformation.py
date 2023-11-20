import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder 
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, type_splitter, log


@dataclass
class DataConfigurationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =  DataConfigurationConfig()
        # self.categorical_colums = []
        # self.numerical_columns = []


    def get_data_transformer_object(self, data):
        try:
            numerical_columns = type_splitter(data, 'float64')
            categorical_columns = type_splitter(data, 'object')
            categorical_columns.remove('Heart_Disease')

            logging.info(f"Numerical columsns:[{numerical_columns}]")
            logging.info(f"Categorical columns:[{categorical_columns}]")

            categorical_pipeline = make_pipeline(OneHotEncoder(sparse_output=False))
            float_pipeline = make_pipeline(StandardScaler())

            logging.info(f"Created data type based pipelines")
            preprocessor = ColumnTransformer([('categorical', categorical_pipeline, categorical_columns),
                                  ('float', float_pipeline, numerical_columns)])
            
            logging.info("Created preprocessor object")
            # categorical_pipeline = make_pipeline(OneHotEncoder(sparse_output=False))
            # integer_pipeline = make_pipeline(FunctionTransformer(log1p), StandardScaler())
            # float_pipeline = make_pipeline(FunctionTransformer(log), StandardScaler())
            # boolean_pipeline = make_pipeline(PCA(n_components=3))

            # preprocessor = ColumnTransformer([('categorical', categorical_pipeline, categorical_),
            #                       ('integer', integer_pipeline, integer_),
            #                       ('float', float_pipeline, float_),
            #                       ('boolean', boolean_pipeline, boolean_)
            #                      ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Obtaining preprocessor object")
            preprocessing_object = self.get_data_transformer_object(train_df)
            
            target_column = "Heart_Disease"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing on train and test data")
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessor to pkl file")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_object)
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e,sys)