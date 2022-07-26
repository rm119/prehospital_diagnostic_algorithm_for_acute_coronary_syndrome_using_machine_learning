# -- import principal modules
import os, sys
import logging
#logging.getLogger().setLevel(logging.DEBUG)
#import warnings
#warnings.filterwarnings("ignore")
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -- import ML modules
from sklearn.ensemble import VotingClassifier


# -- import customized modules
sys.path.append('./utils')
import importdata
import iteration
import evaluation
import imputation
import utils
from global_variables import EXPLANATORY_VARIABLES, SELECTED_VARIABLES

# -- set random seed
# Set a seed value
seed_value= 123
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

n_repeats = 10
n_splits_outer = 5
n_splits_inner = 5

estimators = ['LogisticRegression', 
                            'LinearSVC', 
                            'LGBMClassifier', 
                            'LinearDiscriminantAnalysis',
                            'XGBClassifier',
                            'SVC',
                            'RandomForestClassifier', 
                            'MLPClassifier',  
                            ]

def nestcv_evaluation(target):
    # -- load data
    X, y = importdata.load_traindata(EXPLANATORY_VARIABLES, target)

    for features in [EXPLANATORY_VARIABLES, SELECTED_VARIABLES]:
        
        n_features = len(features)

        if n_features == 43:
            drop_cols = None
        elif n_features == 17:
            drop_cols = list(set(EXPLANATORY_VARIABLES)^set(SELECTED_VARIABLES))

        models = dict()

        for estimatorname in estimators:
            models[estimatorname] = iteration.nestcv(estimatorname, X, y,  
                                                    target=target, 
                                                    drop_cols=drop_cols, 
                                                    n_repeats=n_repeats, 
                                                    n_splits_outer=n_splits_outer, 
                                                    n_splits_inner=n_splits_inner, 
                                                    Tuning=True,
                                                    filename=f'plots/modelevaluation_{target}_{n_features}_features_{estimatorname}.png')
            # -- save models with metadata
            utils.save_data(models[estimatorname], f'models/modelevaluation_{target}_{n_features}_features_{estimatorname}.pkl')


# -- generate the final model
def generate_finalmodel(target):
    """
    Generate a model and evaluate the model using external data sets.
    
    target: str, {'ACS', 'AMI', 'STEMI'}
    
    """
    # -- load data
    X, y = importdata.load_traindata(EXPLANATORY_VARIABLES, target)
    X_ext, y_ext = importdata.load_externaldata(EXPLANATORY_VARIABLES, target)

    for features in [EXPLANATORY_VARIABLES, SELECTED_VARIABLES]:
        
        n_features = len(features)

        if n_features == 43:
            drop_cols = None
        elif n_features == 17:
            drop_cols = list(set(EXPLANATORY_VARIABLES)^set(SELECTED_VARIABLES))

        for estimatorname in estimators:
            results = iteration.parameter_tuning(estimatorname, X, y, target=target, drop_cols=drop_cols, Tuning=True)
            imputator = results['imputator']
            if drop_cols:
               # X_imputed = imputator.fit_transform(X).drop(columns=drop_cols)
                X_ext_imputed = imputator.transform(X_ext).drop(columns=drop_cols)
            else:
               # X_imputed = imputator.fit_transform(X)
                X_ext_imputed = imputator.transform(X_ext)
        
            #evaluation.plots_roc(results['model'], 
            #             X_val=X_imputed
            #             y_val=y, 
            #             fontsize = 17, title=estimatorname,
            #             savefig=f'plots/finalmodel_{target}_{n_features}_features_train_ROC_CURVE_{estimatorname}.png')

            evaluation.plots_roc(results['model'], 
                         X_val=X_ext_imputed,
                         y_val=y_ext, 
                         fontsize = 17, title=estimatorname,
                         savefig=f'plots/finalmodel_{target}_{n_features}_features_externalcohort_ROC_CURVE_{estimatorname}.png')

            results['score_external'] = evaluation.get_scores(results['model'], X_ext_imputed,  
                                                      y_ext, threshold=results['threshold'])
        utils.save_data(results, f'models/finalmodel_{target}_{n_features}_features_{estimatorname}.pkl')
    

def generate_finalmodel_voting(target):

    estimatorname = 'VotingClassifierAll'

    # -- load data
    X, y = importdata.load_traindata(EXPLANATORY_VARIABLES, target)
    X_ext, y_ext = importdata.load_externaldata(EXPLANATORY_VARIABLES, target)

    for features in [EXPLANATORY_VARIABLES, SELECTED_VARIABLES]:
        
        n_features = len(features)

        if n_features == 43:
            drop_cols = None
        elif n_features == 17:
            drop_cols = list(set(EXPLANATORY_VARIABLES)^set(SELECTED_VARIABLES))

        # -- setting submodels
        imputator = imputation.ImputationTransformer()
        
        if drop_cols:
            X_imputed = imputator.fit_transform(X).drop(columns=drop_cols)
            X_ext_imputed = imputator.transform(X_ext).drop(columns=drop_cols)
        else:
            X_imputed = imputator.fit_transform(X)
            X_ext_imputed = imputator.transform(X_ext)

        submodels = list()
        for submodelname in estimators:
            submodel_filename = f'models/finalmodel_{target}_{n_features}_features_{submodelname}.pkl'
            submodel_info = utils.load_data(submodel_filename)
            submodels.append(('pipeline_' + submodelname, submodel_info['model']))

        clf = VotingClassifier(estimators=submodels, voting='soft')
        clf.fit(X_imputed, y)

        results = dict()
        results['model'] = clf
        results['imputator'] = imputator
        results['train_score'] = evaluation.get_scores(clf, X_imputed, y)
        results['threshold'] = evaluation.calc_gmeans(y, clf.predict_proba(X_imputed)[:,1], thresh=None)['threshold']

        # validation with external cohort
        evaluation.plots_roc(results['model'], 
                            X_val=X_ext_imputed, 
                            y_val=y_ext, 
                            fontsize = 17, title=estimatorname,
                            savefig=f'plots/finalmodel_{target}_{n_features}_features_externalcohort_ROC_CURVE_{estimatorname}.png')

        results['score_external'] = evaluation.get_scores(results['model'], X_ext_imputed,  
                                                        y_ext, threshold=results['threshold'])
        # -- save models with metadata
        utils.save_data(results, f'models/finalmodel_{target}_{n_features}_features_{estimatorname}.pkl')


        logging.debug('cross-validation')
        # cross-validation
        cv_score = iteration.outercv(results['model'], X, y, drop_cols=drop_cols, 
                                n_splits_outer=n_splits_outer, n_repeats=n_repeats,
                                filename=f'plots/modelevaluation_{target}_{n_features}_features_{estimatorname}.png')
            
        utils.save_data(cv_score, f'models/modelevaluation_{target}_{n_features}_features_{estimatorname}.pkl')


def print_ncv_scores(target, n_features):
    # -- display nested CV scores  
    models = dict()
    for estimatorname in estimators + ['VotingClassifierAll']:
        models[estimatorname] = utils.load_data(f'models/modelevaluation_{target}_{n_features}_features_{estimatorname}.pkl')

        for label in ['train', 'test']:
            df_score_mean = pd.DataFrame()
            df_score_std = pd.DataFrame()
            for estimatorname in models.keys():
                evalscore = pd.DataFrame({i: models[estimatorname][f'outer_fold_{i}'][f'outer_{label}_score'] for i in range(len(models[estimatorname].keys()))})\
                            .T.agg(['mean','std', 'sem'])

            df_score_mean = df_score_mean.append(evalscore.loc['mean'].rename(estimatorname))
            df_score_std = df_score_std.append(evalscore.loc['std'].rename(estimatorname))

        print(df_score_mean[['ROC AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'PPV', 'NPV']])
        print(df_score_std[['ROC AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1', 'PPV', 'NPV']])


def main():
    logging.info('Evaluation of models with Nested CV.')
    for target in ['ACS', 'AMI', 'STEMI']:
        # -- evaluation models (Nested CV)
        #nestcv_evaluation(target)
        # -- generate final models
        #generate_finalmodel(target)
        # -- generate final VotingClassifier models
        #generate_finalmodel_voting(target)
        print_ncv_scores(target, 17)
        print_ncv_scores(target, 43)


if __name__ == '__main__':
    main()
