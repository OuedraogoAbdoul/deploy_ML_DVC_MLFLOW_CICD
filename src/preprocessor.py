import numpy as np
import pandas as pd

# imblearn
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class TemporalFeaturesExtraction(BaseEstimator, TransformerMixin):
    """Extract year from date

    Args:
        BaseEstimator (_type_): sklean class
        TransformerMixin (_type_): sklean class
    """

    def __init__(self, variables: str):

        self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        X = X.copy()

        X[self.variables] = pd.DatetimeIndex(X[self.variables]).year

        return X


class ExtractZipCode(BaseEstimator, TransformerMixin):
    """Extract or Remove the xxx values from the zipcode

    Args:
        BaseEstimator (_type_): sklean class
        TransformerMixin (_type_): sklean class
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X.zip_code = X.zip_code.str[:3]

        return X


class MissingValuesImputerWarpper(SimpleImputer):
    """Fill in missing values with most frequent

    Args:
        BaseEstimator (_type_): sklean class
        TransformerMixin (_type_): sklean class
    """

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        self.columns = X.columns
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        imputer = imputer.fit(X)

        X = imputer.transform(X)

        X = pd.DataFrame(X, columns=self.columns)
        return X


class ScalerWrapper(MinMaxScaler):
    """Scale all the data

    Args:
        BaseEstimator (_type_): sklean class
        TransformerMixin (_type_): sklean class
    """

    def fit(self, X, y=None):
        self.columns = X.columns.to_list()
        return super().fit(X, y)

    def transform(self, X):
        X = X.copy()

        X = pd.DataFrame(super().transform(X), columns=self.columns)

        return X


class OverUnderSAMPLE(SMOTEENN, SMOTETomek, SMOTE):

    """Oversample and undersample the data

    Args:
        BaseEstimator (_type_): sklean class
        TransformerMixin (_type_): sklean class
    """

    def __init__(self):

        self.y = None

    def fit(self, X, y=None):
        self.y = y

        return self

    def transform(self, X):
        X = X.copy()

        sm = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5, n_jobs=4)

        X_sm, y_sm = sm.fit_resample(X, self.y)

        tl = TomekLinks(sampling_strategy="all", n_jobs=4)

        smtomek = SMOTETomek(
            sampling_strategy="auto", random_state=42, smote=sm, tomek=tl, n_jobs=4
        )

        X, self.y = smtomek.fit_resample(X, self.y)

        return X, self.y
