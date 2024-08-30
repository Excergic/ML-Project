'''
Data output we are getting from data ingestion here in data_transformation
we are applying transformation operation like "handling missing value", "Standardization",
"encoding" and so on ...
'''
import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # to make transformation(encoding) pipeline
from sklearn.impute import SimpleImputer # for missing value handling
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # For pickle file, it has to save some where
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        this function is responsible for data 
        transformation.
        '''
        try:
            numerical_column = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),# For handling missing values
                    ("scaler", StandardScaler()) # for scaling data
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")), # handling missing values
                    ("one_hot_encoder",OneHotEncoder()), # one hot encoding
                    ("scaler", StandardScaler(with_mean=False)) # scaling (for categorical data it is not necessary to standardize data)
                ]

            )
            logging.info(f"categorical columns : {categorical_columns}")
            logging.info(f"numerical columns : {numerical_column}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_column),
                ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor 
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preproceesing_obj = self.get_data_transformer_object() # Needs to converted into pickle file

            target_column_name = "math_score"
            numerical_column = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preproceesing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preproceesing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preproceesing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)


