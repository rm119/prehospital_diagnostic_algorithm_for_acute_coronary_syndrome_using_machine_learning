# Generate Machine Learning models

import sys
import logging
import lightgbm as lgb
import xgboost as xgb
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# --- Import customized modules
#MYDIR = os.getcwd()
#sys.path.append(MYDIR+ '/utils/')
from global_variables import EXPLANATORY_VARIABLES, SELECTED_VARIABLES
import preprocessor
# -- print versions
logging.debug('Python version: {}'.format(sys.version))
logging.debug('Sklearn version: {}'.format(sklearn.__version__))
logging.debug('XGBoost version: {}'.format(xgb.__version__))
logging.debug('LightGBM version: {}'.format(lgb.__version__))

# --- Global variables

classifiers = {'XGBClassifier': xgb.XGBClassifier,
               'LGBMClassifier': lgb.LGBMClassifier,
               'RandomForestClassifier': RandomForestClassifier,
               'LogisticRegression': LogisticRegression,
               'MLPClassifier': MLPClassifier,
               'SVC': SVC,
               'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
               'LinearSVC': LinearSVC
               # 'BernoulliNB': BernoulliNB,
               # 'KNeighborsClassifier': KNeighborsClassifier,
               # 'GaussianNB': GaussianNB,
               # 'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis,
               }

# --- define models


def make_model(estimatorname: str, target: str,
               preprocessor_parm: dict, classifier_parm: dict):
    """
    parameters
    ----------
    estimatorname : str, {'XGBClassifier', 'LGBMClassifier',
             'RandomForestClassifier',
             'LogisticRegression', 'BernoulliNB', 'MLPClassifier',
             'KNeighborsClassifier',
             'GaussianNB','QuadraticDiscriminantAnalysis', 'SVC',
             'LinearDiscriminantAnalysis','LinearSVC'}
    target : str, {'ACS', 'AMI', 'STEMI'}
    preprocessor_parm : dict
        parameters for preprocessor "PreTransformer".
    classifier_parm : dict
        parameters for classifier.
    """

    pipe_step1 = preprocessor.PreTransformer(**preprocessor_parm)
    pipe_step2 = MinMaxScaler()
    pipe_step3 = classifiers[estimatorname](**classifier_parm)

    if estimatorname in ['XGBClassifier', 'RandomForestClassifier', 'LGBMClassifier']:
        model = Pipeline(steps=[('preprocessor', pipe_step1),
                                ('classifier', pipe_step3)])
    elif estimatorname in ['BernoulliNB', 'LogisticRegression', 'SVC', 'MLPClassifier', 'KNeighborsClassifier',
                           'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'GaussianNB',
                           ]:
        model = Pipeline(steps=[('preprocessor', pipe_step1),
                                ('scaler', pipe_step2),
                                ('classifier', pipe_step3)])
    elif estimatorname == 'LinearSVC':
        pipe_step3 = CalibratedClassifierCV(LinearSVC(**classifier_parm))
        model = Pipeline(steps=[('preprocessor', pipe_step1),
                                ('scaler', pipe_step2),
                                ('classifier', pipe_step3)])
    else:
        logging.warn('No model assignment.')
        return None

    return model
