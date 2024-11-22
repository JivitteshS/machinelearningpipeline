import pandas as pd
import pickle
import mlflow
from sklearn.metrics import accuracy_score,confusion_matrix
from mlflow.models import infer_signature
import os
from urllib.parse import urlparse
import yaml

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/JivitteshS/machinelearningpipeline.mlflow"


os.environ["MLFLOW_TRACKING_USERNAME"]="JivitteshS"

os.environ["MLFLOW_TRACKING_PASSWORD"]="2b99e4d1b45979c2beff7cee9a91805fdd76ad4e"


params = yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/JivitteshS/machinelearningpipeline.mlflow")

    ## Load the model from the disk
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)

    accuracy=accuracy_score(y,predictions)

    mlflow.log_metric("accuracy",accuracy)

    print(f"Model accuracy:{accuracy}")


if __name__ == "__main__":
    evaluate(params["data"],params["model"])

                
                        