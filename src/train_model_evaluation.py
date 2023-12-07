import pandas as pd 
import numpy as np 
from read_data import read_params,get_data
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from Data_preprocessing import pre_process
from spliting_data import split_data
from sklearn.model_selection import train_test_split       
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_selection as fs
import json 
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
import joblib

def model_processing(config_path):

    config = read_params(config_path)
    split_data = split_data(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    save_model_path = config["model_dir"]
    criterion = config["estimators"]["random_forest"]["params"]["criterion"]
    max_depth = config["estimators"]["random_forest"]["params"]["max_depth"]
    min_samples_split = config["estimators"]["random_forest"]["params"]["min_samples_split"]
    n_estimators = config["estimators"]["random_forest"]["params"]["n_estimators"]
    column_importance  = config["reports"]["column_importance"]
    scoring = config["model"]["parameter"]["scoring"]
    save_model_path = config["model_dir"]["saved_models"]
    traing_data = pd.read_csv(train_data_path,sep=",", encoding='utf-8')
    test_data = pd.read_csv(test_data_path,sep=",", encoding='utf-8')

    data = traing_data.drop(['Type_num'], axis=1)

    target = traing_data["Type_num"]

    test_target = test_data["Type_num"]

    test_data = test_data.drop(['Type_num'], axis=1)

    

    np.unique(target, return_counts = True)

    np.unique(test_target, return_counts = True)

    Random_forest_model = RandomForestClassifier(n_estimators=100,random_state=999)
    Random_forest_model.fit(data, target)
    random_forest_feature_indices = np.argsort(Random_forest_model.feature_importances_)[::-1][0:10]
    random_forest_column = data.columns[random_forest_feature_indices].values
    rf_feature_importance = Random_forest_model.feature_importances_[random_forest_feature_indices] 

    data_dict = dict(zip(random_forest_column, rf_feature_importance))

    with open(column_importance, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

    mask_df1 = data[random_forest_column]
    test_data = test_data[random_forest_column]

    rf_model = RandomForestClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators
        )
    
    rf_model.fit(mask_df1, target)

    predicted_result = rf_model.predict(test_data)
    accuracy = accuracy_score(test_target, predicted_result)

# F1 score
    f1 = f1_score(test_target, predicted_result)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "Accuracy": accuracy,
            "F1_score": f1,
        }

        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:

        params = {
            "criterion":criterion,
            "max_depth":max_depth,
            "min_samples_split":min_samples_split,
            "n_estimators":n_estimators
        }
        json.dump(params, f, indent = 4)


    joblib.dump(rf_model, save_model_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    model_processing(config_path=parsed_args.config)






    