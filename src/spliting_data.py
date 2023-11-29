import pandas as pd 
import numpy as np 
from read_data import read_params,get_data
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from Data_preprocessing import pre_process
from sklearn.model_selection import train_test_split       
def split_data(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    split_ratio = config["split_data"]["test_size"]
    random_sate = config["base"]["random_state"]
    pre_process_data = pre_process(config_path)

    train, test = train_test_split(
        pre_process_data, 
        test_size=split_ratio, 
        random_state=random_sate
        )
    
    all_columns = train.columns

    df_scaler = preprocessing.MinMaxScaler()
    df_scaler.fit(train)
    train_df= df_scaler.fit_transform(train)
    test_df= df_scaler.fit_transform(test)

    train_df = pd.DataFrame(train_df, columns=all_columns)
    test_df = pd.DataFrame(test_df, columns=all_columns)

    train_df.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test_df.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_data(config_path=parsed_args.config)

