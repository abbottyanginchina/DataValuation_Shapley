from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

def my_accuracy_score(clf, X, y):
    probs = clf.predict_proba(X)
    predictions = np.argmax(probs, -1)
    return np.mean(np.equal(predictions, y))


def my_f1_score(clf, X, y):
    predictions = clf.predict(X)
    if len(set(y)) == 2:
        return f1_score(y, predictions)
    return f1_score(y, predictions, average='macro')


def my_auc_score(clf, X, y):
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    return roc_auc_score(y, true_probs)


def my_xe_score(clf, X, y):
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    true_log_probs = np.log(np.clip(true_probs, 1e-12, None))
    return np.mean(true_log_probs)