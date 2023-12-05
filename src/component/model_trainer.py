import os
import sys
import numpy as np 
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Beginning with model training")
            X_train, y_train, X_test, y_test = train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                # "XG Boost": XGBClassifier()
                }
            
            params = {
                "Logistic Regression":{
                    # 'penalty' : ['l1', 'l2', 'elasticnet', None],
                    # 'C' : np.logspace(-4, 4, 20),
                    # 'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
                    'max_iter' : [10000]
                },
                "Decision Tree": {
                    # "criterion":['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest":{
                    # 'n_estimators':[8,16,32,64,128,256]
                },
                # "XG Boost":{

                # }
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train = y_train,
                                               X_test=X_test,y_test=y_test, 
                                               models=models, param=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted = best_model.predict(X_test)

            # r2_square = r2_score(y_test, predicted)
            return 'so far so good'
        except CustomException as e:
            raise CustomException(e, sys)
               