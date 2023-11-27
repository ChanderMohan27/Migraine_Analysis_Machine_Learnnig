import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    # print(config)
    data_path = config["load_data"]["raw_data_csv"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    return df

