import logging
import numpy as np
import matplotlib.pyplot as plt
import shap
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, average_precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import auc


scoring = {'roc_auc': 'roc_auc',
           'accuracy': 'accuracy',
           'recall': 'recall',
           'specificity': make_scorer(recall_score, pos_label=0),
           'precision':'precision',
           'f1': 'f1',
           'NPV': make_scorer(precision_score, pos_label=0)
           }

def idx_of_the_nearest(data, value):
    idx = np.argmin(np.abs(np.array(data) - value))
    return idx


def calc_gmeans(y_true, y_pred, thresh=None):
    """
    Return the best threshold to maximize the geometrical mean of imbalanced classification.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    gmeans = np.sqrt(tpr * (1-fpr))

    if thresh:
        ix = idx_of_the_nearest(thresholds, thresh)
        logging.debug('Given Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    else:
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        logging.debug('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    results = {"threshold": thresholds[ix],
               "gmeans": gmeans[ix],
               "fpr": fpr[ix],
               "tpr": tpr[ix]}

    return results


def get_scores(model, X, y, threshold=None):
    """
    Calculate the best threshold to maximize the geometrical mean of ROC AUC.
    """
    y_pred_proba = model.predict_proba(X)[:,1]

    if threshold:
        THRESHOLD = threshold
    else:
        THRESHOLD = calc_gmeans(y, y_pred_proba)['threshold']

    y_pred = y_pred_proba>THRESHOLD

    scores = dict()

    # Confusion matrix
    #scores['CM'] = confusion_matrix(y,y_pred)

    scores['ROC AUC'] = roc_auc_score(y, y_pred_proba)

    scores['Accuracy'] = accuracy_score(y, y_pred)

    # Sensitivity, Recall, TPR
    scores['Sensitivity'] = recall_score(y, y_pred)

    # Specificity, TNR
    scores['Specificity'] = recall_score(y, y_pred, pos_label=0)

    # Precision (PPV)
    scores['PPV'] = precision_score(y, y_pred)

    # NPV
    scores['NPV'] = precision_score(y, y_pred, pos_label=0)

    scores['F1'] = f1_score(y,y_pred)

    scores['THRESHOLD'] = THRESHOLD

    return scores


def cv_scores(model, X, y, n_splits=10):
    cv = StratifiedKFold(n_splits=n_splits)
    scores = cross_validate(model, X, y, cv=cv,
                            scoring = scoring,
                            return_train_score=True)
    return scores



def plots_roc(clf, X_train=None, y_train=None, X_test=None, y_test=None, savefig=None,
                X_val=None, y_val=None, fontsize = 17, title=None):
    # AUC curve
    fig, ax = plt.subplots(figsize=(3, 3))
    if X_train is not None:
        RocCurveDisplay.from_predictions(
            y_train, clf.predict_proba(X_train)[:, 1],
            ax=ax, name='Training', color='C0')
    if X_test is not None:
        RocCurveDisplay.from_predictions(
            y_test, clf.predict_proba(X_test)[:, 1],
            ax=ax, name='Test', color='C1')
    if X_val is not None:
        RocCurveDisplay.from_predictions(
            y_val, clf.predict_proba(X_val)[:, 1],
            ax=ax, name='Validation', color='C2')
    ax.set_ylabel('Sensitivity', fontsize=fontsize)
    ax.set_xlabel('1-Specificity', fontsize=fontsize)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    plt.title(title)
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=150)
    else:
        plt.show()

    
    
def plots_shapsummary(clf, X_train, savefig=None):
    # SHAP
    explainer = shap.Explainer(clf.named_steps['classifier'])
    shap_values = explainer(clf.named_steps['preprocessor'].transform(X_train))
    feature_names = clf.named_steps['preprocessor'].transform(X_train).columns
    if savefig:
        show = False
    else:
        show = True

    shap.summary_plot(shap_values, plot_size=(5, 10), show=show,
                      feature_names=feature_names, max_display=len(feature_names))  # , plot_type='violin')
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()



def score_ci_boot( y_true, y_pred_proba, metrics='roc_auc', n_bootstraps = 1000, alpha=0.05, threshold=0.5):
    """
     Calculate confidence intervals using bootstrap method. Supported metrics are
    Sensitivity, ROC AUC, Specificity, Averaged Precision, F1, Accuracy, Precision, and NPV.
    
    alpha: float, default=0.05
        significance level
    """
    supported_metrics = ['sensitivity','specificity','f1','roc_auc','accuracy', 'ap', 'precision', 'NPV']
    if metrics not in supported_metrics:
        logging.error('metrics name has to be:{}'.format(','.join(supported_metrics)))
        return None
    
    np.random.seed(123)
    rng=np.random.RandomState(123)

    # define threhold

    y_pred = y_pred_proba>threshold
    
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))

        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        
        if metrics == 'roc_auc':
            score = roc_auc_score(y_true[indices], y_pred_proba[indices])
        elif metrics == 'ap':
            score = average_precision_score(y_true[indices], y_pred_proba[indices])
        elif metrics == 'sensitivity':
            score = recall_score(y_true[indices], y_pred[indices], pos_label=1)
        elif metrics == 'specificity':
            score = recall_score(y_true[indices], y_pred[indices], pos_label=0)
        elif metrics == 'f1':
            score = f1_score(y_true[indices], y_pred[indices])
        elif metrics == 'accuracy':
            score = accuracy_score(y_true[indices], y_pred[indices])
        elif metrics == 'precision':
            score = precision_score(y_true[indices], y_pred[indices])
        elif metrics == 'NPV':
            score = accuracy_score(y_true[indices], y_pred[indices])
            
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

   # 95% c.i.
   # confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
   # confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    confidence_lower = sorted_scores[int(alpha/2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1-alpha/2) * len(sorted_scores))]

    return confidence_lower, confidence_upper


def help():
    pass


if __name__ == '__main__':
    help()