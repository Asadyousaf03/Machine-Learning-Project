#utils have the common things, that we are trying to import and use in the entire project.
import os
import sys

from src.logger import logging
import pickle
import dill
from sklearn.metrics import accuracy_score, classification_report  # Modified import
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_classification_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        logging.info("yaha tak to ma a gay hu... shid age masla ho.") 
        for i in range(len(list(models))):  # Iterating directly over dictionary items
            model = list(models.values())[i]
            # Hyper-param Tuning code.
        #     para=param[list(models.keys())[i]]

        #     gs = GridSearchCV(model,para,cv=3)
        #     gs.fit(X_train,y_train)
        #  #   model.fit(X_train, y_train)
        #     model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # For classification, use accuracy as an evaluation metric
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # You can also include other classification metrics like precision, recall, and F1-score
           # classification_metrics = classification_report(y_test, y_test_pred, output_dict=True)

            # Store the metrics in the report dictionary
            report[list(models.keys())[i]] = test_model_score
            logging.info(f"yar {i} loop ma koi masla nahi ha. ") 

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)