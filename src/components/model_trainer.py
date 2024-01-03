
import os
import sys
from dataclasses import dataclass

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_classification_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, threshold=0.6):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Splitting training and test input data.... Done")

            models = {
                "Gaussian Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(criterion ="entropy", max_depth = 4),
                "Random Forest": RandomForestClassifier(n_estimators=30),
                "Logistic Regression": LogisticRegression(max_iter=1000),
            }


            # params = {
            # "Decision Tree": {
            #     'criterion': ['entropy', 'gini'],
            #                 },
            # "Random Forest": {
            #     'n_estimators': [8, 16, 32, 64, 128, 256]
            # },
            # "Gaussian Naive Bayes": {
            #     'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            # },
            # "Logistic Regression": {
            #     'C': [0.001, 0.01, 0.1, 1, 10, 100],
            #     'penalty': ['l1', 'l2'],
            #     'solver': ['lbfgs', 'liblinear', 'saga']
            # }
            # }


            logging.info("yaha tak chal raha hai... age dekho")


            model_report = evaluate_classification_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info("evaluation done..")
            best_model_name, best_model_accuracy = self.get_best_model(model_report)

            if best_model_accuracy < threshold:
                raise CustomException("No best model found with a satisfactory accuracy.")

            logging.info(f"Best model: {best_model_name} with accuracy of {best_model_accuracy:.2%}")

            best_model = models[best_model_name]

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            #classification_metrics = classification_report(y_test, predicted, output_dict=True)

            return accuracy
                 #  "best_model_name": best_model_name,
                 # "best_model_accuracy": best_model_accuracy,
                 #"classification_metrics": classification_metrics
            

        except Exception as e:
            raise CustomException(e, sys)


    def get_best_model(self, model_report):
        best_model_accuracy = max(sorted(model_report.values()))
        best_model_name = next((name for name, acc in model_report.items() if acc == best_model_accuracy), None)

        return best_model_name, best_model_accuracy















