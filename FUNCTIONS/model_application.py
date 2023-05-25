import config as c

import numpy as np
import pandas as pd
import time

# Create sparse matrices to run the scikit multilearn algorithms
from scipy.sparse import lil_matrix

# Models
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import (
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset,
)
from skmultilearn.adapt import MLkNN

# Scoring Metrics
from sklearn.metrics import accuracy_score, hamming_loss


class ModelApplication:
    def __init__(self, X_train, X_train_pca, y_train, X_test, X_test_pca, y_test):
        self.X_train = X_train
        self.X_train_pca = X_train_pca
        self.y_train = y_train
        self.X_test = X_test
        self.X_test_pca = X_test_pca
        self.y_test = y_test

    def sparse_matrix(self):

        X_train_pca = lil_matrix(self.X_train_pca).toarray()
        y_train = lil_matrix(self.y_train).toarray()
        X_test_pca = lil_matrix(self.X_test_pca).toarray()
        y_test = lil_matrix(self.y_test).toarray()

        return X_train_pca, y_train, X_test_pca, y_test

    def model_application(self, model, X_train, X_test):

        # Start Time
        start_time = time.time()
        # Problem Transform
        # with Gaussian Naive Bayes Classifier
        model_classifier = model(GaussianNB())
        # train
        model_classifier.fit(X_train, self.y_train)
        # predict
        predictions = model_classifier.predict(X_test)
        # accuracy
        accuracy = accuracy_score(self.y_test, predictions)
        accuracy = np.round_(accuracy * 100, decimals=3)
        # hamming loss
        h_loss = hamming_loss(self.y_test, predictions)
        h_loss = np.round_(h_loss, decimals=4)
        # End Time
        end_time = np.round(((time.time() - start_time) / 60), 2)

        return end_time, accuracy, h_loss

    def run_model(self, X_train, X_test):

        model_dict = {
            "Binary Relevance": BinaryRelevance,
            "Classifier Chains": ClassifierChain,
            "Label Powerset": LabelPowerset,
        }

        result_df = pd.DataFrame()
        for key, value in model_dict.items():
            end_time, accuracy, h_loss = self.model_application(value, X_train, X_test)
            row = {
                "Classifier": key,
                "Run time": end_time,
                "Accuracy %": accuracy,
                "Hamming Loss": h_loss,
            }
            rowdf = pd.DataFrame.from_records([row])
            result_df = pd.concat([result_df, rowdf], ignore_index=True, axis=0)

        return result_df

    def run_mlknn(self, X_train, y_train, X_test, y_test):

        # Start Time
        start_time = time.time()
        # Adaptive Algorithm
        model_classifier = MLkNN(k=3)
        # train
        model_classifier.fit(X_train, y_train)
        # predict
        predictions = model_classifier.predict(X_test)
        # accuracy
        accuracy = accuracy_score(y_test, predictions)
        accuracy = np.round_(accuracy * 100, decimals=3)
        # hamming loss
        h_loss = hamming_loss(y_test, predictions)
        h_loss = np.round_(h_loss, decimals=4)
        # End Time
        end_time = np.round(((time.time() - start_time) / 60), 2)
        result_df_2 = pd.DataFrame(
            {
                "Classifier": "Multi-Label kNN",
                "Run time": end_time,
                "Accuracy %": accuracy,
                "Hamming Loss": h_loss,
            },
            index=[0],
        )

        return result_df_2, predictions

    def result_df(self, df1, df2):
        return pd.concat([df1, df2], ignore_index=True, axis=0).reset_index(drop=True)

    def model_analysis(self, X_train, y_train, X_test, y_test):
        df_1 = self.run_model(X_train, X_test)
        df_2, predictions = self.run_mlknn(X_train, y_train, X_test, y_test)
        result_df = self.result_df(df_1, df_2)
        return result_df, predictions


def data_predict(y, predictions_df):

    # Map the labels to their name
    labels = y.columns.tolist()
    lables_dict = {key: count for count, key in enumerate(labels)}
    # Turn predictions into readable list of lists
    my_predictions = predictions_df.rows.tolist()
    # Translate the values to their keys
    get_key = [k for k, v in lables_dict.items() if v in my_predictions[0]]
    # Save Predictions to new CSV
    test_predictions = pd.DataFrame(my_predictions, columns=["BuildingID", "Floor"])
    test_predictions.to_csv(f"{c.loc_prd}/testData_predictions.csv", index=False)

    return get_key
