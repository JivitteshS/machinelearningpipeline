import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os
import mlflow
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/JivitteshS/machinelearningpipeline.mlflow"


os.environ["MLFLOW_TRACKING_USERNAME"]="JivitteshS"

os.environ["MLFLOW_TRACKING_PASSWORD"]="2b99e4d1b45979c2beff7cee9a91805fdd76ad4e"

def hyperparameter_tuning(X_train, y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_job=1)
    grid_search.fit(X_train, y_train)

    return grid_search

## Load the parameters from params.yaml

params=yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/JivitteshS/machinelearningpipeline.mlflow")

    ## start the MLflow run
    with mlflow.start_run():
        # Train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
        signature=infer_signature(X_train,y_train)

        ##Defining hyperparameter grid

        param_grid = {
            'n_estimators': [100,200],
           'max_depth': [5,10,None],
            'min_samples_split':[2,5],
            'min_samples_leaf':[1,2]
        }

        # perform the hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

        ## get the best model
        best_model = grid_search.best_estimator_

        # predict abd evaluate the model
        y_pred = best_model.predict(X_test)

        accuracy_score = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy score: {accuracy_score}")






