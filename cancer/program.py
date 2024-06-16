import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrices(actual, pred):
    accuracy = metrics.accuracy_score(actual,pred)
    precision = metrics.precision_score(actual, pred)
    recall = metrics.recall_score(actual, pred)
    return accuracy, precision, recall

def logRegModel():
    model = LogisticRegression()
    return model

def dtModel():
    model = DecisionTreeClassifier()
    return model

def rfModel():
    params = {'max_depth': 6, 'random_state':2020}
    model = RandomForestClassifier(**params)
    return model, params

def create_split_features(pred_var):
    train_X = train[pred_var]
    train_Y = train.diagnosis
    test_X = test[pred_var]
    test_Y = test.diagnosis
    return train_X, train_Y, test_X, test_Y

if __name__ =="__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(1010)

    data = pd.read_csv('data.csv')
    #data.drop("Unnamed: 32", axis=1, inplace=True)
    data.drop("id", axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    
    pred_var_mean = [x for x in data.columns if "_mean" in x]
    pred_var_se = [x for x in data.columns if "_se" in x]
    pred_var_worst = [x for x in data.columns if "_worst" in x]

    train, test = train_test_split(data, test_size=0.3, random_state = 1010)

    train_X, train_Y, test_X, test_Y = create_split_features(pred_var_mean)
    print (train_X)
    print (train_Y)
    print (test_X)
    print (test_Y)
    

    with mlflow.start_run():
        # model = logRegModel()
        # model = dtModel()
        model, params = rfModel()
        model.fit(train_X, train_Y)

        predictions = model.predict(test_X)

        (accuracy, precision, recall) = eval_metrices(test_Y, predictions)

        mlflow.log_metric("Acc", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

        mlflow.log_params(params)

        predictions = model.predict(train_X)
        signature = infer_signature(train_X, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = "Mean Run",
            registered_model_name= "Random Forest",
            signature = signature
        )
