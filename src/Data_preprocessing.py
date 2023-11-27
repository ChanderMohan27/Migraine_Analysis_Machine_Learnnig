import pandas as pd 
import numpy as np 
from read_data import read_params,get_data
import argparse

def pre_process(config_path):
    df = get_data(config_path)
    print(df.head())

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    pre_process(config_path=parsed_args.config)    
    

