# machinelearningpipeline



The dvc stage add command is used to define stages in ML or data pipeline. These stages represent steps like data preprocessing , model training or evaluation.

Three important idependet conpoenents

ml pipeline
prreprocessing --> training --> evaluation

to define things in pipeline there is DVC Stages

Example Stage 1 --> Stage 2 --> Stage 3


ML Flow Experiments
DVC data versioning


Preprocessing pipeline

dvc stage add -n preprocess \
-p preprocess.input,preprocess.output \
-d src/preprocess.py -d data/raw/data.csv \
-o data/preprocessed/data.csv \
python src/preprocess.py


training pipeline

dvc stage add -n train \
-p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
-d src/train.py -d data/raw/data.csv \
-o models/model.pkl \
python src/train.py

dvc stage add -n evaluate \
-d src/evaluate.py --force \
-d models/model.pkl \
-d data/raw/data.csv \
python src/evaluate.py 


Form Dagshub --> Remote --> Data
Add a Dagshub DVC remote

Setup credentials

