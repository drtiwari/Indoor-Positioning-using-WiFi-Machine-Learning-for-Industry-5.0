import pandas as pd
import config as c
import time
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, hamming_loss

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import metrics


from skmultilearn.adapt import MLkNN

from FUNCTIONS.data_prepration import DataPreprocess


def run_prediction_analysis(data_file, targets):
    # Start Time
    start_time = time.time()

    # LOAD DATASET
    Data = pd.read_csv(f"{c.loc_idat}/{data_file}.csv")

    # INSTANTIATE DATA PREPARATION CLASS
    C2 = DataPreprocess(df=Data)

    # FEATURES ENGINEERING
    df = C2.clean_data()

    # FEATURES AND TARGET SPLITTING
    X, y = C2.preprocess_data(df, targets)

    # LABELS MAPPING
    labels = y.columns.tolist()
    lables_dict = {key: count for count, key in enumerate(labels)}

    # SCALE
    # Scale Data with Standard Scaler
    scaler = StandardScaler()
    # Fit the data
    scaler.fit(X)
    X = scaler.transform(X)

    # PCA
    pca = PCA(0.95)
    # Fit
    pca.fit(X)
    X_pca = pca.transform(X)
    print(
        "PCA RESULTS",
        "-----------",
        "TOTAL PCA: {}".format(pca.n_components_),
        "TOTAL VARIANCE: {}".format(round(pca.explained_variance_ratio_.sum(), 3)),
        " ",
        sep="\n",
    )

    # SPARSE MATRIX
    X_pca = lil_matrix(X_pca).toarray()
    y = lil_matrix(y).toarray()

    # PREDICT
    MLKNN_1_classifier = MLkNN(k=1)
    # train
    MLKNN_1_classifier.fit(X_pca, y)
    predictions = MLKNN_1_classifier.predict(X_pca)

    # ACCURACY
    accuracy = accuracy_score(y, predictions)
    accuracy = np.round_(accuracy * 100, decimals=3)

    # HAMMING LOSS
    h_loss = hamming_loss(y, predictions)
    h_loss = np.round_(h_loss, decimals=4)

    # TRANSLATE

    my_predictions = predictions.rows.tolist()
    # translate the values to their keys.
    get_keys = [k for k, v in lables_dict.items() if v in my_predictions[0]]

    # Save Predictions to new CSV
    test_predictions = pd.DataFrame(my_predictions, columns=targets)
    test_predictions.to_csv(f"{c.loc_prd}/{data_file}_predictions.csv", index=False)

    # End Time
    end_time = np.round(((time.time() - start_time) / 60), 2)

    print(
        "PREDICTION RESULTS",
        "-----------",
        "RUN TIME: {}".format(end_time),
        "ACCURACY: {}".format(accuracy),
        "HAMMING LOSS: {}".format(h_loss),
        "PREDICTION: {}".format(get_keys),
        sep="\n",
    )

    return end_time, accuracy, h_loss, get_keys


def neural_network_regression(X_train, X_test, y_train, y_test):

    model = keras.Sequential(
        [
            layers.Flatten(),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(2, activation="linear"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=[
            metrics.RootMeanSquaredError(),
            metrics.CosineSimilarity(axis=1),
        ],
    )
    epochs = 5
    val_data = (X_test, y_test)
    result = model.fit(
        X_train,
        y_train,
        validation_data=val_data,
        epochs=epochs,
    )

    return result


def run_prediction_analysis_NN(data_file, targets):
    # Start Time
    start_time = time.time()

    # LOAD DATASET
    Data = pd.read_csv(f"{c.loc_idat}/{data_file}.csv")

    # INSTANTIATE DATA PREPARATION CLASS
    C = DataPreprocess(df=Data)

    # FEATURES ENGINEERING
    df = C.clean_data()

    # FEATURES AND TARGET SPLITTING
    X, y = C.preprocess_data(df, targets)

    # LABELS MAPPING
    labels = y.columns.tolist()
    lables_dict = {key: count for count, key in enumerate(labels)}

    # SCALE
    # Scale Data with Standard Scaler
    scaler = StandardScaler()
    # Fit the data
    scaler.fit(X)
    X = scaler.transform(X)

    # TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = C.split_data(X, y, False)

    # NEURAL NETWORK REGRESSION
    result = neural_network_regression(X_train, X_test, y_train, y_test)

    # End Time
    end_time = np.round(((time.time() - start_time) / 60), 2)

    return end_time, result
