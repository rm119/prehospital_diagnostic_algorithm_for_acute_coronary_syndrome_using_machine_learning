import sys
import os
from get_feature_names_for_sklearn1_0_2 import get_feature_names
from sklearn.neighbors import KNeighborsRegressor
# -- estimators for imputation
#from sklearn.kernel_approximation import Nystroem
#from sklearn.linear_model import BayesianRidge, Ridge
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import sklearn
import pandas as pd
import numpy as np
#import warnings
# warnings.filterwarnings('ignore')


# --- Import customized modules
#MYDIR = os.getcwd()
#sys.path.append(MYDIR + '/utils/')
sys.path.append('./utils')

if sklearn.__version__ < '1.1':
    NEWSKLERAN = False
else:
    NEWSKLERAN = True


class ImputationTransformer(TransformerMixin, BaseEstimator):
    """
        Constructs a transformer for imputation.

    parameters
    ----------
    random_seed: int
        Random seed used for IterativeImputer

    Examples
    --------
    X : DataFrame
    >>> transformer = ImputationTransformer()
    >>> transformer.fit(X)
    >>> transformer.transform(X)
    """

    def __init__(self, random_seed=2679, sample_posterior=False, estimator=KNeighborsRegressor()):
        self.imputations = dict()
        self.random_seed = random_seed
        self.sample_posterior = sample_posterior
        self.estimator = estimator

    def _imp_interviews(self, random_seed):
        """
        Imputation among interviews' columns (except interview16 and 17)
        """
        key_name = 'interviews'
        self.imputations[key_name] = dict()
        self.imputations[key_name]['cat'] = True
        self.imputations[key_name]['scale'] = False
        self.imputations[key_name]['features'] = [
            f'interview{i}' for i in range(1, 19) if i not in [16, 17]]
        self.imputations[key_name]['params'] = {"max_iter": 100,
                                                "min_value": 0,
                                                "max_value": 10,
                                                "sample_posterior": self.sample_posterior,
                                                "random_state": random_seed
                                                }
        self.imputations[key_name]['params']['estimator'] = self.estimator
        imputator = ('imp', IterativeImputer(
            **self.imputations[key_name]['params']))
        # if NEWSKLERAN:
        #    int_convertor = ('to_int', FunctionTransformer(
        #        to_int, feature_names_out='one-to-one'))
        # else:
        #    int_convertor = ('to_int', FunctionTransformer(to_int))
        self.imputations[key_name]['pipe'] = Pipeline(
            steps=[imputator])  # , int_convertor])

    def _imp_medical_histories(self):
        """
        medical histories and interviews 16 & 17 are imputed with a fixed value.
        """
        key_name = 'medicalHistories'
        self.imputations[key_name] = dict()
        self.imputations[key_name]['cat'] = True
        self.imputations[key_name]['scale'] = False
        self.imputations[key_name]['features'] = [f'medical_history{i}' for i in range(1, 11) if i not in [7, 8]]\
            + [f'interview{i}' for i in [16, 17]]
        self.imputations[key_name]['params'] = {
            "strategy": 'constant', "fill_value": self.medical_history_filledvalue}

        imputator = ('imp', SimpleImputer(
            **self.imputations[key_name]['params']))

        self.imputations[key_name]['pipe'] = Pipeline(
            steps=[imputator])

    def _imp_blood_pressure(self):
        """
        Imputation for blood pressures using kNN
        """
        key_name = 'bloodPressure'
        self.imputations[key_name] = dict()
        self.imputations[key_name]['cat'] = False
        self.imputations[key_name]['scale'] = True
        self.imputations[key_name]['features'] = [
            'blood_pressure_max', 'blood_pressure_min']
        self.imputations[key_name]['params'] = {"n_neighbors": 5}
        self.imputations[key_name]['pipe'] = Pipeline(steps=[
            ('imp', KNNImputer(**self.imputations[key_name]['params'])),
        ])

    def _imp_vitals(self):
        """
        Vital signes are imputed with median values
        """
        key_name = 'vitalSigns'
        self.imputations[key_name] = dict()
        self.imputations[key_name]['cat'] = False
        self.imputations[key_name]['scale'] = True
        self.imputations[key_name]['features'] = [
            'pulse', 'body_temperature', 'spo2', 'breathing']
        self.imputations[key_name]['params'] = {"strategy": 'median'}
        self.imputations[key_name]['pipe'] = Pipeline(steps=[
            ('imp', SimpleImputer(**self.imputations[key_name]['params'])),
        ])

    def set_imputator(self):
        self.transformer = ColumnTransformer(
            transformers=[(key, value['pipe'], value['features'])
                          for key, value in self.imputations.items()],
            remainder='passthrough'
        )

    def _set_interview_default(self, X, y=None):
        """
        Interview definition changed from (3:yes, 2:no) to (2:yes, 0:no)
        """
        interview_map = {3: 2, 1: np.nan, 2: 0}
        for col in [f'interview{coli}' for coli in range(1, 18)]:
            if col in X.columns.to_list():
                X[col] = X[col].map(interview_map)
        self.interview_vmin = min(interview_map.values())  # 0
        self.interview_vmax = max(interview_map.values())  # 2
        return X

    def _set_medical_history_default(self, X, y=None):
        """
        Interview definition changed from (3:yes, 2:no) to (2:yes, 0:no)
        """
        medical_history_map = {3: 2, 1: np.nan, 2: 0}
        for col in [f'medical_history{i}' for i in range(1, 11) if i not in [7, 8]]:
            if col in X.columns.to_list():
                X[col] = X[col].map(medical_history_map)
        self.medical_history_filledvalue = min(
            medical_history_map.values())  # 0
        return X

    def _adjust_interview_minmax(self, X, y=None):
        """
        Round up/down imputed values to fit within the value range of each column.
        """
        interview_vmin = {
            f'interview{i}': self.interview_vmin for i in range(1, 18)}
        interview_vmax = {
            f'interview{i}': self.interview_vmax for i in range(1, 18)}
        interview_vmin['interview18'] = 1
        interview_vmax['interview18'] = 10
        for col in [f'interview{coli}' for coli in range(1, 19)]:
            if col in X.columns.to_list():
                X[col] = X[col].apply(round)
                X.loc[X[col] < interview_vmin[col], col] = interview_vmin[col]
                X.loc[X[col] > interview_vmax[col], col] = interview_vmax[col]
        return X

    def fit(self, X, y=None):
        X = self._set_interview_default(X.copy(deep=True))
        X = self._set_medical_history_default(X.copy(deep=True))
        self._imp_interviews(self.random_seed)
        self._imp_medical_histories()
        self._imp_blood_pressure()
        self._imp_vitals()
        self.set_imputator()
        logging.debug("Fitting")
        self.transformer.fit(X.copy(deep=True))

        if not NEWSKLERAN:
            columns_after_imputation = get_feature_names(
                self.transformer)
            convert_x_colnames = {f'x{i}': X.columns[i] for i in range(
                len(columns_after_imputation))}
            for i, col in enumerate(columns_after_imputation):
                if col in convert_x_colnames.keys():
                    columns_after_imputation[i] = convert_x_colnames[col]
                if '__' in col:
                    columns_after_imputation[i] = col.split('__')[-1]
            self.feature_names_out = columns_after_imputation

        return self

    def transform(self, X, y=None):
        check_is_fitted(self.transformer, 'transform')
        logging.debug("Transforming")
        X = self._set_interview_default(X.copy(deep=True))
        X = self._set_medical_history_default(X.copy(deep=True))
        X_transformed = pd.DataFrame(self.transformer.transform(X))
        if NEWSKLERAN:
            X_transformed.columns = [col_org.split(
                '__')[-1] for col_org in self.transformer.get_feature_names_out()]
        else:
            X_transformed.columns = self.feature_names_out

        X_transformed.index = X.index
        X_transformed = self._adjust_interview_minmax(X_transformed)
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X, y=y)
