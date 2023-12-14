# Migraine Analysis



## Overview

 In this machine learning project, the primary focus is on establishing a robust MLOps pipeline, integrating it with a CI/CD pipeline, and leveraging the Data Version Control (DVC) framework. The goal is to streamline and automate the end-to-end machine learning lifecycle.

 ## Dataset

 The dataset used in this project is taken from Kaggle. You can download the dataset [here](https://www.kaggle.com/datasets/ranzeet013/migraine-dataset).



 The migraine dataset provides detailed insights into migraine headaches, covering demographic details like age, specifics of each episode (duration, frequency), and characteristics of pain (location, intensity). It also includes accompanying symptoms such as nausea and vomiting, offering a holistic view. Additionally, sensory and neurological aspects like phonophobia, photophobia, and visual disturbances are considered. This dataset holds significant potential for research, diagnosis, and treatment strategies, serving as a valuable resource for uncovering patterns and correlations in the world of migraines.


## CI-CD Pipline and Model Selection 

To develop this model using a CI/CD pipeline, we utilized Cookiecutter to establish the project structure. For Data Version Control, the DVC framework along with dvc-gdrive was employed. After structuring the project, we implemented a dvc.yaml file, segregating the code into distinct modules like load_data.py, Data_preprocessing.py, and train_model.py. The chosen model is a Random Forest, suitable for this classification problem given the imbalanced nature of the data. To address the imbalance, we aggregated some target values. Additionally, feature engineering was applied to identify and utilize the top 10 relevant features.


## Results Storage

We utilize JSON files to automatically store important results from our model evaluation. The following metrics are stored:

- **F1 Score:** The F1 score is a measure of a model's accuracy that considers both precision and recall. You can find the F1 score results in the `report/scores.json` file.

- **Accuracy:** The accuracy metric represents the overall correctness of the model. Our accuracy results are stored in the `report/scores.json` file.

These JSON files serve as valuable references for assessing the performance of our model over time.

