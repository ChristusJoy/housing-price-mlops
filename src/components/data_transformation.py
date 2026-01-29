"""
Manages data transformation (preprocessing, sealing, imputing) for training.
Inputs: DataIngestionArtifact, DataValidationArtifact, DataTransformationConfig
Outputs: DataTransformationArtifact (preprocessor, transformed data)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a data transformer object for the data.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            logging.info("Transformers Initialized: Numeric (Median Imputer + StandardScaler)")

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            logging.info("Transformers Initialized: Categorical (Constant Imputer + OneHotEncoder)")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            cat_features = self._schema_config['categorical_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("Numeric", numeric_transformer, num_features),
                    ("Categorical", categorical_transformer, cat_features)
                ],
                remainder='passthrough'
            )

            logging.info("Final Pipeline Ready!!")
            return preprocessor

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting feature engineering")
            if {'YrSold', 'YearBuilt'}.issubset(df.columns):
                df['HouseAge'] = df['YrSold'] - df['YearBuilt']
            if {'YrSold', 'YearRemodAdd'}.issubset(df.columns):
                df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
            
            bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
            if all(col in df.columns for col in bath_cols):
                df['TotalBathrooms'] = (
                    df['FullBath'] + 0.5 * df['HalfBath'] + 
                    df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
                )
            
            area_cols = ['GrLivArea', 'TotalBsmtSF']
            if all(col in df.columns for col in area_cols):
                df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']

            if 'GarageArea' in df.columns:
                df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
            if 'TotalBsmtSF' in df.columns:
                df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)

            logging.info("Feature engineering completed successfully")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            input_feature_train_df = self._create_features(input_feature_train_df)
            input_feature_test_df = self._create_features(input_feature_test_df)

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys) from e
