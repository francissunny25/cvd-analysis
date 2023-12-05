import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predictor(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            processed_data = preprocessor.transform(features)
            predictions = model.predict(processed_data)
            return predictions
        
        except Exception as e:
            raise CustomException
class CustomData:
    def __init__(self, General_Health: str, Checkup: str, Age_Category: str, Exercise: str, 
                 Skin_Cancer: str, Other_Cancer: str, Depression: str, Diabetes: str, Arthritis: str, 
                 Sex: str, Smoking_History:str, Height_cm: float, Weight_kg: float, BMI: float, 
                 Alcohol_Consumption: int, Fruit_Consumption: int, Green_Vegetables_Consumption: int, 
                 FriedPotato_Consumption: int 
                 ):
          self.General_Health = General_Health
          self.Checkup = Checkup
          self.Age_Category = Age_Category
          self.Exercise = Exercise
          self.Skin_Cancer = Skin_Cancer
          self.Other_Cancer = Other_Cancer
          self.Depression = Depression
          self.Diabetes = Diabetes
          self.Arthritis = Arthritis
          self.Sex = Sex
          self.Smoking_History = Smoking_History
          self.Height_cm = Height_cm
          self.Weight_kg = Weight_kg
          self.BMI = BMI
          self.Alcohol_Consumption = Alcohol_Consumption
          self.Fruit_Consumption = Fruit_Consumption
          self.Green_Vegetables_Consumption = Green_Vegetables_Consumption
          self.FriedPotato_Consumption = FriedPotato_Consumption
    def get_data_as_data_frame(self):
         try:
              custom_data_input = {
                   'General_Health': [self.General_Health],
                   'Checkup': [self.Checkup],
                   'Age_Category': [self.Age_Category],
                   'Exercise': [self.Exercise],
                   'Skin_Cancer': [self.Skin_Cancer],
                   'Other_Cancer': [self.Other_Cancer],
                   'Depression': [self.Depression],
                   'Diabetes': [self.Diabetes],
                   'Arthritis': [self.Arthritis],
                   'Sex': [self.Sex],
                   'Smoking_History': [self.Smoking_History],
                   'Height_(cm)': [self.Height_cm],
                   'Weight_(kg)': [self.Weight_kg],
                   'BMI': [self.BMI],
                   'Alcohol_Consumption': [self.Alcohol_Consumption],
                   'Fruit_Consumption': [self.Fruit_Consumption],
                   'Green_Vegetables_Consumption': [self.Green_Vegetables_Consumption],
                   'FriedPotato_Consumption': [self.FriedPotato_Consumption]
                   }
              return pd.DataFrame(custom_data_input)
         except Exception as e:
            raise CustomException(e, sys)
              

