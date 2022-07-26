from get_feature_names_for_sklearn1_0_2 import get_feature_names
import logging
import sklearn
import pandas as pd
import numpy as np
import warnings
# warnings.filterwarnings('ignore')
import os
import sys

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
#from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.validation import check_is_fitted

# --- Import customized modules
#MYDIR = os.getcwd()
#sys.path.append(MYDIR+ '/utils/')
sys.path.append('./utils')


if sklearn.__version__ < '1.1':
    NEWSKLERAN = False
else:
    NEWSKLERAN = True


class ScalingTransformer(TransformerMixin, BaseEstimator):
    """
        Constructs a StandardScaler transformer for numeric features.
    Non-numeric featurs are return untouched.

    Examples
    --------
    X : DataFrame
    >>> sc = ScalingTransformer()
    >>> sc.fit(X)
    >>> sc.transform(X)
    """

    def __init__(self):
        self.defaultNumericalFeatures = ['blood_pressure_max',
                                         'blood_pressure_min',
                                         'pulse',
                                         'body_temperature',
                                         'spo2',
                                         'breathing',
                                         'age']

    def set_transformer(self):
        self.transformer = ColumnTransformer(
            transformers=[('scale', StandardScaler(),
                           self.numericalFeatures)],
            remainder='passthrough'
        )

    def get_feature_names_out(self, name):
        return self.transformer.get_feature_names_out(self.transformer.feature_names_in_)

    def fit(self, X, y=None):
        self.numericalFeatures = [
            col for col in X.columns if col in self.defaultNumericalFeatures]
        self.set_transformer()
        self.transformer.fit(X)
        self.feature_names_out = [col_org.split(
            '__')[-1] for col_org in self.transformer.get_feature_names_out()]
        #print("ScalingTransformer N={}".format(len(self.feature_names_out)), self.feature_names_out)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self.transformer, 'transform')
        X_transformed = self.transformer.transform(X)
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)
            X_transformed.columns = self.feature_names_out
            X_transformed.index = X.index

        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=y)


class CategoricalTransformer(TransformerMixin, BaseEstimator):
    """
        Constructs a Onehot encoding transformer for categorical features.
    Numeric featurs are return untouched.

    Examples
    --------
    X : DataFrame
    >>> ct = CategoricalTransformer()
    >>> ct.fit(X)
    >>> ct.transform(X)
    """

    def __init__(self):
        self.defaultCategoricalFeatures = [f'interview{i}' for i in range(1, 18)]\
            + [f'medical_history{i}' for i in [1, 2, 3, 4, 5, 6, 9, 10]]

    def set_transformer(self):

        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore', drop='if_binary')

        self.transformer = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.CategoricalFeatures)],
            remainder='passthrough'
        )

    def set_unknown_to_nan(self, X):
        unknown_values = 1
        for col in self.CategoricalFeatures:
            X[col] = X[col].replace(unknown_values, np.nan)
        #no_values = 0
        # for col in  self.CategoricalFeatures:
        #    X[col] = X[col].replace(no_values, np.nan)
        return X

    def remove_nan(self, X):
        for col in self.CategoricalFeatures:
            if col + '_nan' in X.columns:
                X.drop(columns=[col + '_nan'], inplace=True)
        return X

    def get_feature_names_out(self, name):
        return self.transformer.get_feature_names_out(self.transformer.feature_names_in_)

    def fit(self, X, y=None):
        self.CategoricalFeatures = [
            col for col in X.columns if col in self.defaultCategoricalFeatures]
        X = self.set_unknown_to_nan(X)
        self.set_transformer()
        self.transformer.fit(X)
        self.feature_names_out = [col_org.split(
            '__')[-1] for col_org in self.transformer.get_feature_names_out()]
        #print("CategoricalTransformer N={}".format(len(self.feature_names_out)), self.feature_names_out)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self.transformer, 'transform')
        X = self.set_unknown_to_nan(X)
        X_transformed = self.transformer.transform(X)
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)
            X_transformed.columns = self.feature_names_out
            X_transformed.index = X.index
        X_transformed = self.remove_nan(X_transformed)

        return X_transformed


class OrdinalTransformer(TransformerMixin, BaseEstimator):
    """
        Constructs a Ordinal encoding transformer for categorical features
    ('interview' and 'medical history' columns).
    Numeric featurs are return untouched.

    Examples
    --------
    X : DataFrame
    >>> ot = OrdinalTransformer()
    >>> ot.fit(X)
    >>> ot.transform(X)
    """

    def __init__(self):
        self.defaultCategoricalFeatures = [f'interview{i}' for i in range(1, 19)]\
            + [f'medical_history{i}' for i in [1, 2, 3, 4, 5, 6, 9, 10]]

    def set_transformer(self):

        categorical_transformer = OrdinalEncoder(
            handle_unknown='ignore')

        self.transformer = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.CategoricalFeatures)],
            remainder='passthrough'
        )

    def get_feature_names_out(self, name):
        return self.transformer.get_feature_names_out(self.transformer.feature_names_in_)

    def fit(self, X, y=None):
        self.CategoricalFeatures = [
            col for col in X.columns if col in self.defaultCategoricalFeatures]
        self.set_transformer()
        self.transformer.fit(X)
        self.feature_names_out = [col_org.split(
            '__')[-1] for col_org in self.transformer.get_feature_names_out()]
        #print("OrdinalTransformer N={}".format(len(self.feature_names_out)), self.feature_names_out)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self.transformer, 'transform')
        X_transformed = self.transformer.transform(X)
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)
            X_transformed.columns = self.feature_names_out
            X_transformed.index = X.index
        return X_transformed


class PreTransformer(TransformerMixin, BaseEstimator):
    """
        Constructs a transformer for StandardScaler, OneHotEncoder, and
    VarianceThreshold.

    parameters
    ----------
    onehot : bool
        if True, OnehotEncoder is applied, in addition to StandardScaler and
        VarianceThreshold. if False, OnehotEncoder is NOT applied for
        categorical features.

    Examples
    --------
    X : DataFrame
    >>> pt = PreTransformer()
    >>> pt.fit(X)
    >>> pt.transform(X)
    """

    def __init__(self, onehot=True):
        self.onehot = onehot

    def set_transformer(self, X):
        if self.Categorical:
            self.transformer = Pipeline(steps=[
                ('scaling', ScalingTransformer()),
                ('preprocessor', CategoricalTransformer()),
                ('filter', VarianceThreshold(threshold=(.95 * (1 - .95))))
            ])

        else:
            self.transformer = Pipeline(steps=[
                ('scaling', ScalingTransformer()),
                #('preprocessor', OrdinalTransformer()),
                ('filter', VarianceThreshold(threshold=(.95 * (1 - .95))))])

    def fit(self, X, y=None):
        if self.onehot:
            self.Categorical = True
            _ct = CategoricalTransformer()
            _ct.CategoricalFeatures = [
                col for col in X.columns if col in _ct.defaultCategoricalFeatures]
            if len(_ct.CategoricalFeatures) == 0:
                self.Categorical = False
            else:
                logging.debug("OneHotEncoder is set.")
        else:
            self.Categorical = False
        self.set_transformer(X)
        self.transformer.fit(X)
        self.feature_names_out = self.transformer.named_steps['filter'].get_feature_names_out(
        )
        #print("PreTransformer N={}".format(len(self.feature_names_out)), self.feature_names_out)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self.transformer, 'transform')
        X_transformed = self.transformer.transform(X)
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)
            X_transformed.columns = self.feature_names_out
            X_transformed.index = X.index

        return X_transformed
