import logging
import numpy as np
import optuna
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import xgboost as xgb
#-- import customized modules
from preprocessor import PreTransformer

# -- Define the objective function
n_splits = 5
N_TRIALS = 100
HIDDEN_LAYER_SIZES = [(64, ), (32, ), (16, ), (8,), (4,),
                      (64, 32), (32, 16), (16, 8), (8, 4), (4, 2)
                    ]

def setup_preprocessor(trial):
    # -- Proprocessing
    # (a) Onehot Encoording for Categorical variables or not
    onehot = trial.suggest_categorical('onehot', [True, False])

    if onehot == True:
        preprocessor = PreTransformer(onehot=True)
    elif onehot == False:
        preprocessor = PreTransformer(onehot=False)

    return preprocessor


def get_param_dist_lr(trial, solver):
    params = {'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
             'C': trial.suggest_float('C', 1e-1, 1e2, log=True),
             'random_state': trial.suggest_int('random_state', 1, 10),
             'max_iter': trial.suggest_int('max_iter',100, 10000),
    }
    if solver in ['newton-cg',  'lbfgs', 'sag', '0']:
        params['solver'] = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'sag'])
        params['penalty'] = trial.suggest_categorical('penalty', ['l2', 'none'])
    elif solver in ['liblinear', '1']:
        params['solver'] = trial.suggest_categorical('solver', ['liblinear'])
        params['penalty'] = trial.suggest_categorical('penalty', ['l1','l2'])
    elif solver in ['saga', '2']:
        params['solver'] = trial.suggest_categorical('solver', ['saga'])
        params['penalty'] = trial.suggest_categorical('penalty', ['elasticnet', 'none'])
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

    return params

# -- Optuna for Logistic Regression
#      There are several combinations of parameters are banned, so
#      try to generate several objectives:
def objective_variable_solver(solver, X, y):
    def objective(trial):
        cv = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
        preprocessor = setup_preprocessor(trial)
        scaler = MinMaxScaler()

        # -- Instantiate estimator model
        params = get_param_dist_lr(trial, solver)
        estimator = LogisticRegression(**params)

        # -- Make a pipeline
        pipeline = make_pipeline(preprocessor, scaler, estimator)

        # -- Evaluate the score by cross-validation
        score = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv)
        score_mean = score.mean() # calculate the mean of scores
        return score_mean

    return objective


def tuning_logistic_regression(X, y, n_trials=N_TRIALS):
    final_trials = []
    for solver in list('012'):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_variable_solver(solver, X, y), n_trials=n_trials)
        final_trials.append(study)

    for i in range(len(final_trials)):
        print(f'Logistic ver.0{i}', final_trials[i].best_trial.params, final_trials[i].best_trial.value)

    return final_trials


# -- Optuna for SVC

def get_param_dist(trial, estimatorname):
    if estimatorname == 'SVC':
        params = {'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
             'probability':  trial.suggest_categorical('probability', [True]),
             'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
             #'random_state': trial.suggest_int('random_state', 0, 10),
             'gamma': trial.suggest_float('gamma', 1e-3, 1e3, log=True),
             }
    elif estimatorname == 'LinearSVC':
        params = {'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
             'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
             'penalty': trial.suggest_categorical('penalty', ['l2', 'l1']),
             'dual': trial.suggest_categorical('dual', [False]),
             'max_iter': trial.suggest_int('max_iter',100, 10000),
        }
    elif estimatorname == 'RandomForestClassifier':
        params = {'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
             'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),#, 'log_loss']),
             'max_depth': trial.suggest_int('max_depth', 2, 8),
             'max_features': trial.suggest_float('max_features', 0, 1),
             'max_samples': trial.suggest_float('max_samples', 0.1, 0.9),
             'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
             #'random_state': trial.suggest_int('random_state', 0, 10),
             }
    elif estimatorname == 'MLPClassifier':
        params = {
             #'early_stopping': trial.suggest_categorical('early_stopping', [True]),
             'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', HIDDEN_LAYER_SIZES),
             #'random_state': trial.suggest_int('random_state', 0, 10),
             'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.005),
             'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
             'validation_fraction': trial.suggest_float('validation_fraction', 0., 0.3),
             'max_iter': trial.suggest_int('max_iter',100, 10000),
             'alpha': trial.suggest_float('alpha', 1e-2, 1e2, log=True),
             }
    elif estimatorname == 'KNeighborsClassifier':
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 200),

        }
    elif estimatorname == 'LGBMClassifier':
        params = {
            'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
            'objective': trial.suggest_categorical('objective', ['binary']),
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.005),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            #'random_state': trial.suggest_int('random_state', 0, 10),
        }
    elif estimatorname == 'XGBClassifier':
        params = {
                'verbosity': 0,
                'objective': trial.suggest_categorical('objective', ['binary:logistic']),
                # use exact for small dataset.
                'tree_method': 'exact',
                # defines booster, gblinear for linear functions.
                'booster': trial.suggest_categorical('booster', ['gblinear','gbtree', 'dart']), 
                # L2 regularization weight.
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                # L1 regularization weight.
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1, log=True),
                # sampling ratio for training data.
                'subsample': trial.suggest_float('subsample', 0.60, 0.95),
                # sampling according to each tree.
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 0.95),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 0.5),
                'random_state': trial.suggest_int('random_state', 0, 10),
            }
        if params['booster'] in ['gbtree', 'dart']:
            # maximum depth of the tree, signifies complexity of the tree.
            params['max_depth'] = trial.suggest_int('max_depth', 2, 8)
            # minimum child weight, larger the term more conservative the tree.
            params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10, log=True)
            params['max_bin'] = trial.suggest_int('max_bin', 2, 33)
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
            # defines how selective algorithm is.
            params['gamma'] = trial.suggest_float('gamma', 1e-3, 5.0, log=True)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

        if params['booster'] == 'dart':
            params['sampling_method'] = trial.suggest_categorical('sampling_method', ['uniform'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 1e-3, 1.0, log=True)
            params['skip_drop'] = trial.suggest_float('skip_drop', 1e-3, 1.0, log=True)
    elif estimatorname == 'QuadraticDiscriminantAnalysis':
        params = {
            'reg_param': trial.suggest_float('reg_param', 1e-8, 10, log=True),
        }
    elif estimatorname == 'LinearDiscriminantAnalysis':
        params = {
            'solver': trial.suggest_categorical('solver', ['lsqr']),
            'shrinkage': trial.suggest_categorical('shrinkage', ['auto', 1e-3, 1e-2, 1e-1, 0, 1, None]),
        }

    return params


def objectives(estimatorname, X, y):
    def objective(trial):

        preprocessor = setup_preprocessor(trial)

        # -- Instantiate estimator model
        params = get_param_dist(trial, estimatorname)
        scaler = MinMaxScaler()

        if estimatorname == 'SVC':
            estimator = SVC(**params)
        elif estimatorname == 'LinearSVC':
            estimator = CalibratedClassifierCV(LinearSVC(**params))
        elif estimatorname == 'RandomForestClassifier':
            estimator = RandomForestClassifier(**params)
        elif estimatorname == 'LGBMClassifier':
            estimator = lgb.LGBMClassifier(**params)
        elif estimatorname == 'XGBClassifier':
            params['scale_pos_weight'] = (y==0).sum()/(y==1).sum()
            estimator = xgb.XGBClassifier(**params)
        elif estimatorname == 'MLPClassifier':
            estimator = MLPClassifier(**params)
        elif estimatorname == 'KNeighborsClassifier':
            estimator = KNeighborsClassifier(**params)
        elif estimatorname == 'QuadraticDiscriminantAnalysis':
            estimator = QuadraticDiscriminantAnalysis(**params)
        elif estimatorname == 'LinearDiscriminantAnalysis':
            estimator = LinearDiscriminantAnalysis(**params)

        # -- Make a pipeline
        if estimatorname in ['SVC', 'LinearSVC', 'MLPClassifier', 'KNeighborsClassifier', 'QuadraticDiscriminantAnalysis', 'LinearDiscriminantAnalysis', 'ANN']:
            pipeline = make_pipeline(preprocessor, scaler, estimator)
        else:
            pipeline = make_pipeline(preprocessor, estimator)

        # -- Evaluate the score by cross-validation
        cv = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
        score = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv)
        score_mean = score.mean() # calculate the mean of scores
        return score_mean

    return objective


def tuning(estimatorname, X, y, n_trials=N_TRIALS):
    logging.debug(estimatorname)
    if estimatorname == 'LogisticRegression':
        study = tuning_logistic_regression(X, y, n_trials=n_trials)
    else:
        study = optuna.create_study(direction='maximize')
        study.optimize(objectives(estimatorname, X, y), n_trials=n_trials)
        print('best CV scores', study.best_trial.value)
        print('Best parameters', study.best_trial.params)
    return study


def help():
    print('# -- How to use')
    print('# -- LogisticRegression, SVC, RandomForestClassifier, LinearSVC, lightGBM, XGBoost, MLPC')
    print('           tuning("LogisticRegression", X, y)')
    print('           tuning("SVC", X, y)')
    print('           tuning("RandomForestClassifier", X, y)')
    print('           tuning("LinearSVC", X, y)')
    print('           tuning("LGBMClassifier", X, y)')
    print('           tuning("XGBClassifier", X, y)')
    print('           tuning("MLPClassifier", X, y)')


if __name__ == '__main__':
    help()
