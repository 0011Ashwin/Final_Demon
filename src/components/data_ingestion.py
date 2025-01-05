## Firstly we import important librariesfrom python module
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## import modules from src.components for DataTransformation
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

## import modules from src,components for ModelTraining
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# configuration class for ingestion_path_config
@dataclass
## Giving the configuration of input of raw_data_file
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

## Data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

## Start data ingestion process with initiate_data_ingestion
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            ## Reading the data file 
            df=pd.read_csv('notebook\\data\\stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            ## The data split into train and test data 80% train and 20% test data
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            ## Reading of train data from file path
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            ## Reading of test data from file path
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            ## Returning the file_path of data_set
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))