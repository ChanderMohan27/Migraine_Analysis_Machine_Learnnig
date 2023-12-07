import pandas as pd 
import numpy as np 
from read_data import read_params,get_data
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

def converter(number):
    if number<=18:
        new_number = 0
    elif (number>18 and number<=40):
        new_number = 1
    else:
        new_number = 2
    return new_number
def target_coverter(number):
    if number<=4:
        new_number = 0
    else:
        new_number = 1
    return new_number

def pre_process(config_path):
    df = get_data(config_path)
    config = read_params(config_path)
    cleaned_data = config["load_data"]["cleaned_data_csv"]
    encoder = LabelEncoder()
    df['Type_num'] = encoder.fit_transform(df['Type'])
    df = df.drop(columns='Type')

    df["Age"] = df["Age"].map(converter)
    df["Type_num"] = df["Type_num"].apply(target_coverter)
    df.to_csv(cleaned_data, sep=",", index=False, encoding="utf-8")

    return df 

    # all_columns = data.columns
    # df_scaler = preprocessing.MinMaxScaler()
    # df_scaler.fit(data)
    # mask_df1= df_scaler.fit_transform(data)
    # mask_df1 = pd.DataFrame(mask_df1, columns=all_columns)






if __name__=="__main__": 
    args = argparse.ArgumentParser() 
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    pre_process(config_path=parsed_args.config)    
    
    

