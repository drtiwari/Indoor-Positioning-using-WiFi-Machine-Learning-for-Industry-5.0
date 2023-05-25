import config as c

from IPython.display import display
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class DataPreparation:
    def __init__(self, df):
        self.df = df

    # EXPLORATORY DATA ANALYSIS
    def exp_data_analysis(
        self,
        structure=True,
        description=True,
        information=True,
        column_names=True,
        missing_data_per=True,
    ):

        # Check the structure of the data after it's loaded
        if structure:
            # (print the number of rows and columns).
            num_rows, num_cols = self.df.shape
            print("Number of columns: {}".format(num_cols))
            print("Number of rows: {}".format(num_rows))

        # Check the statistics of the data per columns
        if description:
            display(self.df.describe())

        # Check dataframe datatypes
        if information:
            display(self.df.info())

        # Check the columns names
        if column_names:
            col_names = self.df.columns.values
            display(col_names)

        # Check for missing values
        if missing_data_per:
            missing_values_count = self.df.isnull().sum()
            # print(missing_values_count)

            # how many total missing values do we have?
            total_cells = np.product(self.df.shape)
            total_missing = missing_values_count.sum()

            # percent of data that is missing
            missing_percent = (total_missing / total_cells) * 100

            print("Percent of missing data = {}%".format(missing_percent))

    # UNIQUE ATTRIBUTES IN DATA
    def unique_values(self):

        unique_floors = np.sort(self.df["FLOOR"].unique())
        unique_bldgs = np.sort(self.df["BUILDINGID"].unique())
        unique_spaceid = np.sort(self.df["SPACEID"].unique())
        unique_rpos = np.sort(self.df["RELATIVEPOSITION"].unique())
        unique_users = np.sort(self.df["USERID"].unique())
        print("Unique Floors : {}".format(unique_floors))
        print("Unique Buildings : {}".format(unique_bldgs))
        print("Unique Space IDs : {}".format(unique_spaceid))
        print("Unique Relative Positions : {}".format(unique_rpos))
        print("Unique Users : {}".format(unique_users))

    # DATA AND USER PLOT
    def map_n_details(self):

        self.df.plot(
            kind="scatter", x="LONGITUDE", y="LATITUDE", alpha=0.4, figsize=(9, 6)
        )
        plt.savefig(f"{c.loc_ana}/data_map")
        self.df.plot(
            kind="scatter",
            x="LONGITUDE",
            y="LATITUDE",
            alpha=0.4,
            figsize=(10, 6),
            c="USERID",
            cmap=plt.get_cmap("jet"),
            colorbar=True,
            sharex=False,
        )
        plt.savefig(f"{c.loc_ana}/user_map")

    # CORRELATION, HISTOGRAM, SCATTER PLOTS
    def cor_hist_scat(self):
        # plot the correlations between the WAP features
        corr_matrix = self.df.corr()
        fig = plt.figure(figsize=(15, 15))
        sns.heatmap(corr_matrix, xticklabels=False, yticklabels=False)
        plt.savefig(f"{c.loc_ana}/corr_matrix_plot")

        # plot the histograms of the attributes
        self.df.iloc[:, 520:529].hist(bins=50, figsize=(15, 15))
        plt.savefig(f"{c.loc_ana}/attribute_histogram_plot")

        # plot the scattermatrix of the attributes
        attributes = [
            "BUILDINGID",
            "FLOOR",
            "LATITUDE",
            "LONGITUDE",
            "SPACEID",
            "RELATIVEPOSITION",
        ]
        scatter_matrix(self.df[attributes], figsize=(15, 15))
        plt.savefig(f"{c.loc_ana}/scatter_matrix")


class DataPreprocess:
    def __init__(self, df):
        self.df = df

    # FEATURES ENGINEERING
    def clean_data(self):
        """
        INPUT: trainingData DataFrame
        OUTPUT: Trimmed and cleaned trainingData DataFrame
        """

        # FEATURE TRANSFORMATION
        # Reverse the representation for the values.
        # 100=0 and the values range from 0-105 (weakest to strongest).
        self.df.iloc[:, 0:520] = np.where(
            self.df.iloc[:, 0:520] <= 0,
            self.df.iloc[:, 0:520] + 105,
            self.df.iloc[:, 0:520] - 100,
        )

        # FEATURE TRIMMING
        # Remove selected columns.
        columns_removed = ["USERID", "PHONEID", "TIMESTAMP"]
        for col in columns_removed:
            self.df.drop(col, axis=1, inplace=True)

        df = self.df.copy()

        return df

    # FEATURES AND TARGET SPLITTING
    def preprocess_data(self, df, targets):
        """
        INPUT: Cleaned trainingData DataFrame
        OUTPUT: trainingData as Features and Targets
        """

        global X
        global y

        # Split the data set into features and targets
        # FEATURES:  WAP001 -- WAP520
        X = df.drop(
            [
                "LONGITUDE",
                "LATITUDE",
                "FLOOR",
                "BUILDINGID",
                "SPACEID",
                "RELATIVEPOSITION",
            ],
            axis=1,
        )
        # TARGETS: Floor and BuildingID
        y = df[targets]

        # create Dummies for the targets to feed into the model
        # Qualitative variables / discrete variables / indicator variables/ design variables/ basis variables
        y = pd.get_dummies(data=y, columns=targets)

        return X, y

    # TRAIN-TEST DATA SPLIT
    def split_data(self, X, y, sv_dt):
        # TO AVOID OVERFITTING: Split the training data into training and testing sets
        global X_train
        global X_test
        global y_train
        global y_test

        self.X = X
        self.y = y

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, shuffle=True
        )

        # Show the results of the split
        print("Training set has {} samples.".format(X_train.shape[0]))
        print("Testing set has {} samples.".format(X_test.shape[0]))

        if sv_dt:
            X_train.to_csv(f"{c.loc_rdat}/X_train.csv", index=False)
            X_test.to_csv(f"{c.loc_rdat}/X_test.csv", index=False)
            y_train.to_csv(f"{c.loc_rdat}/y_train.csv", index=False)
            y_test.to_csv(f"{c.loc_rdat}/y_test.csv", index=False)

        return X_train, X_test, y_train, y_test

    # DATA STANDARDIZATION
    def standarization_transformation_data(self, X_train, X_test):
        # Scale Data with Standard Scaler
        scaler = StandardScaler()

        # Fit only the training set
        # this will help us transform the validation data
        scaler.fit(X_train)

        # Apply transform to both the training set and the test set.
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test


class PCAnalysis:
    def __init__(self, var, X_train, X_test):
        self.var = var
        self.X_train = X_train
        self.X_test = X_test

    # DIMENSIONALITY REDUCTION
    def pca_analysis(self):
        # Apply PCA while keeping 95% of the variation in the data
        pca = PCA(self.var)

        # Fit only the training set
        pca.fit(self.X_train)

        # Apply PCA transform to both the training set and the test set.
        X_train_pca = pca.transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)

        print("Number of PCA Components = {}.".format(pca.n_components_))
        # print(pca.n_components_)
        print(
            "Total Variance Explained by PCA Components = {}.".format(
                round(pca.explained_variance_ratio_.sum(), 3)
            )
        )
        # Show the results of the split
        print("Training set has {} samples.".format(X_train_pca.shape[0]))
        print("Testing set has {} samples.".format(X_test_pca.shape[0]))
        # print(pca.explained_variance_ratio_.sum())

        return pca, X_train_pca, X_test_pca

    # PCA SCREE PLOT
    def pca_scree_plot(self, pca):
        num_components = len(pca.explained_variance_ratio_)
        ind = np.arange(num_components)
        vals = pca.explained_variance_ratio_

        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)
        cumvals = np.cumsum(vals)
        ax.bar(ind, vals)
        ax.plot(ind, cumvals)
        for i in range(num_components):
            ax.annotate(
                r"%s%%" % ((str(vals[i] * 100)[:4])),
                (ind[i] + 0.2, vals[i]),
                va="bottom",
                ha="center",
                fontsize=12,
            )

        ax.xaxis.set_tick_params(width=0)
        ax.yaxis.set_tick_params(width=2, length=12)

        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained (%)")
        plt.title("Explained Variance Per Principal Component")
        plt.savefig(f"{c.loc_ana}/pca_scree_plot")
