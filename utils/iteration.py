import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
import imputation
import evaluation
import tuning
from generate_models import make_model

# -- set random seed
# Set a seed value
seed_value= 123
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def nestcv(estimatorname, X, y,  target, n_repeats=10, n_splits_outer=10, n_splits_inner=5, 
               drop_cols=None, Tuning=True,
               filename=None):

    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=n_repeats, random_state=seed_value)
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed_value)

    imputator = imputation.ImputationTransformer()

    # for ROC plot
    fig, ax = plt.subplots(figsize=(4,4))
    tprs = {'train': [], 'test': []}
    aucs = {'train': [], 'test': []}
    mean_fpr = {'train':  np.linspace(0, 1, 100),
                'test': np.linspace(0, 1, 100)}

    model_candidates = dict()
    for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        logging.debug('outer fold{}'.format(outer_fold))
        X_train_outer, X_val_outer = X.iloc[train_index], X.iloc[test_index]
        y_train_outer, y_val_outer = y.iloc[train_index], y.iloc[test_index]

        inner_models = dict()
        for innder_fold, (train_index_inner, test_index_inner) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
            inner_models[innder_fold] = dict()
            logging.debug(': inner fold{}'.format(innder_fold))
            X_train_inner, X_val_inner = X_train_outer.iloc[
                train_index_inner], X_train_outer.iloc[test_index_inner]
            y_train_inner, y_val_inner = y_train_outer.iloc[
                train_index_inner], y_train_outer.iloc[test_index_inner]
            # -- imputation
            logging.debug('imputation for inner fold{}'.format(innder_fold))
            X_train_inner_imputated = imputator.fit_transform(X_train_inner)
            X_val_inner_imputated = imputator.transform(X_val_inner)
            if drop_cols:
                X_train_inner_imputated = X_train_inner_imputated.drop(columns=drop_cols)
                X_val_inner_imputated = X_val_inner_imputated.drop(columns=drop_cols)
            # -- tuning 
            if Tuning:
                n_trials = 100
                logging.debug('tuning for inner fold{}'.format(innder_fold))
                study = tuning.tuning(estimatorname, X_train_inner_imputated, y_train_inner,
                                      n_trials=n_trials)
                #print('best score = {:.3f}'.format(study.best_value))

                if estimatorname == 'LogisticRegression': 
                    params = study[np.argmax([s.best_value for s in study])].best_params
                else:
                    params = study.best_params
                    if estimatorname == 'XGBoost': 
                        params["scale_pos_weight"] = (y_train_inner==0).sum()/(y_train_inner==1).sum()
                if 'onehot' in params.keys():
                    params_onehot = params.pop('onehot')
                else:
                    params_onehot = False
                inner_model = make_model(estimatorname, target, preprocessor_parm={'onehot': params_onehot}, classifier_parm=params)

            else:
                inner_model = make_model(estimatorname, target, preprocessor_parm={'onehot': False}, classifier_parm={})

            logging.debug('fitting a model for inner fold{}'.format(innder_fold))
            inner_model.fit(X_train_inner_imputated, y_train_inner)
            inner_models[innder_fold]['model'] = inner_model
            inner_models[innder_fold]['imputator'] = imputator
            inner_models[innder_fold]['inner_val_score'] = evaluation.get_scores(inner_model, X_val_inner_imputated, y_val_inner)
            logging.debug(">> inner val score {}".format(inner_models[innder_fold]['inner_val_score']))

        logging.debug('Selecting best models for each fold')
        best_score_index = np.argmax([inner_models[innder_fold]['inner_val_score']['ROC AUC'].mean() for innder_fold in inner_models.keys()])
        best_outer_model = inner_models[best_score_index]
        logging.debug('Best model:')
        logging.debug(best_outer_model['model'])
        # -- imputation
        logging.debug('Calculating validation score for each fold')
        X_train_outer_imputated = best_outer_model['imputator'].transform(X_train_outer.copy())
        X_val_outer_imputated = best_outer_model['imputator'].transform(X_val_outer.copy())
        if drop_cols:
            X_train_outer_imputated = X_train_outer_imputated.drop(columns=drop_cols)
            X_val_outer_imputated = X_val_outer_imputated.drop(columns=drop_cols)
        outer_train_score = evaluation.get_scores(best_outer_model['model'], X_train_outer_imputated, y_train_outer)
        outer_val_score = evaluation.get_scores(best_outer_model['model'], X_val_outer_imputated, y_val_outer)
        best_outer_model['outer_val_score'] = outer_val_score
        best_outer_model['outer_train_score'] = outer_train_score

        logging.debug('Saving model scores per fold for plotting')
        for xsub, ysub, label in zip([X_train_outer_imputated, X_val_outer_imputated], [y_train_outer, y_val_outer], ['train', 'test']):
            fpr, tpr, thresholds = roc_curve(ysub, best_outer_model['model'].predict_proba(xsub)[:,1])
            fold_roc_auc = auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr[label], fpr, tpr)
            interp_tpr[0] = 0.0
            tprs[label].append(interp_tpr)
            aucs[label].append(fold_roc_auc)

        best_outer_model['tprs'] = tprs
        best_outer_model['mean_fpr'] = mean_fpr
        best_outer_model['aucs'] = aucs

        model_candidates[f'outer_fold_{outer_fold}'] = best_outer_model

    logging.debug('Plotting ROC')
    for label, color in zip(['train', 'test'], ['C0', 'C1']):
        mean_tpr = np.mean(tprs[label], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr[label], mean_tpr)
        std_auc = np.std(aucs[label])
        ax.plot(
                  mean_fpr[label],
                  mean_tpr,
                  color=color,
                  label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                  lw=2,
                  alpha=0.8)
        std_tpr = np.std(tprs[label], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr[label],
            tprs_lower,
            tprs_upper,
            color=color,
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    ax.legend(loc="lower right")
    plt.title(estimatorname)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.show()

    return model_candidates
    
    


def outercv(model, X, y, n_splits_outer, n_repeats, drop_cols=None, filename=None, title=None):
    
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits_outer, n_repeats=n_repeats, random_state=seed_value)

    imputator = imputation.ImputationTransformer()

    # -- for ROC plot
    fig, ax = plt.subplots(figsize=(4,4))
    tprs = {'train': [], 'test': []}
    aucs = {'train': [], 'test': []}
    mean_fpr = {'train':  np.linspace(0, 1, 100),
                'test': np.linspace(0, 1, 100)}
    
    outer_scores = dict()
    for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        outer_scores[f'outer_fold_{outer_fold}'] = dict()
        logging.debug('Outer fold{}'.format(outer_fold))
        X_train_outer, X_val_outer = X.iloc[train_index], X.iloc[test_index]
        y_train_outer, y_val_outer = y.iloc[train_index], y.iloc[test_index]

        # -- imputation
        logging.debug('imputation for outer fold{}'.format(outer_fold))
        X_train_outer_imputated = imputator.fit_transform(X_train_outer)
        X_val_outer_imputated = imputator.transform(X_val_outer)

        if drop_cols:
            X_train_outer_imputated = X_train_outer_imputated.drop(columns=drop_cols)
            X_val_outer_imputated = X_val_outer_imputated.drop(columns=drop_cols)
        
        logging.debug('fitting a model for outer fold{}'.format(outer_fold))
        model.fit(X_train_outer_imputated, y_train_outer)
        
        outer_scores[f'outer_fold_{outer_fold}']['model'] = model
        outer_scores[f'outer_fold_{outer_fold}']['imputator'] = imputator
        outer_scores[f'outer_fold_{outer_fold}']['outer_train_score'] = evaluation.get_scores(model, X_train_outer_imputated, y_train_outer)
        outer_scores[f'outer_fold_{outer_fold}']['outer_val_score'] = evaluation.get_scores(model, X_val_outer_imputated, y_val_outer)
        
        for xsub, ysub, label in zip([X_train_outer_imputated, X_val_outer_imputated], [y_train_outer, y_val_outer], ['train', 'test']):
            fpr, tpr, thresholds = roc_curve(ysub, model.predict_proba(xsub)[:,1])
            fold_roc_auc = auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr[label], fpr, tpr)
            interp_tpr[0] = 0.0
            tprs[label].append(interp_tpr)
            aucs[label].append(fold_roc_auc)

        outer_scores[f'outer_fold_{outer_fold}']['tprs'] = tprs
        outer_scores[f'outer_fold_{outer_fold}']['mean_fpr'] = mean_fpr
        outer_scores[f'outer_fold_{outer_fold}']['aucs'] = aucs

    logging.debug('plotting ROC CURVE')
    for label, color in zip(['train', 'test'], ['C0', 'C1']):
        mean_tpr = np.mean(tprs[label], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr[label], mean_tpr)
        std_auc = np.std(aucs[label])
        ax.plot(mean_fpr[label],
                  mean_tpr,
                  color=color,
                  label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                  lw=2,
                  alpha=0.8)
        std_tpr = np.std(tprs[label], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr[label],
            tprs_lower,
            tprs_upper,
            color=color,
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    ax.legend(loc="lower right")
    plt.title(title)
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.show()


    return outer_scores
    
    
def parameter_tuning(estimatorname, X, y, target=None, drop_cols=None, Tuning=True):
  
    imputator = imputation.ImputationTransformer()
    results = dict()
    # -- imputation
    logging.debug(': imputation ')
    X_imputed = imputator.fit_transform(X)
    if drop_cols:
        X_imputed = X_imputed.drop(columns=drop_cols)
    # -- tuning 
    if Tuning:
        logging.debug(': tuning')
        study = tuning.tuning(estimatorname, X_imputed, y, n_trials=100)
        if estimatorname == 'LogisticRegression': 
            params = study[np.argmax([s.best_value for s in study])].best_params
        else:
            params = study.best_params
            if estimatorname == 'XGBoost': 
                params["scale_pos_weight"] = (y==0).sum()/(y==1).sum()
            if estimatorname in ['RandomForestClassifier', 'LGBMClassifier', 'XGBClassifier']:
                params_onehot = False
            else:
                params_onehot = params.pop('onehot')
        model = make_model(estimatorname, target, preprocessor_parm={'onehot': params_onehot}, classifier_parm=params)
    else:
        model = make_model(estimatorname, target, preprocessor_parm={'onehot': False}, classifier_parm={})
    
    # -- fit model
    model.fit(X_imputed, y)
    results['model'] = model
    results['imputator'] = imputator
    results['train_score'] = evaluation.get_scores(model, X_imputed, y)
    results['threshold'] = evaluation.calc_gmeans(y, results['model'].predict_proba(X_imputed)[:,1], thresh=None)['threshold']

    # -- extract feature name
    results['processed_featurename'] = results['model']['preprocessor'].feature_names_out
    results['processed_train_data'] = results['model']['preprocessor'].transform(X_imputed)

    return results
