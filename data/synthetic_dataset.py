import models.Nets as nets
from utils.parameters import args_parser
import numpy as np
from scipy.stats import logistic

args = args_parser()
hidden_units = [] # Empty list in the case of logistic regression.
train_size = 100

def create_synthetic_dataset():
    d, difficulty = 50, 1
    num_classes = 2
    tol = 0.03
    target_accuracy = 0.7
    important_dims = 5
    clf = nets.return_model(args.model, solver='liblinear', hidden_units=tuple(hidden_units))
    _param = 1.0
    for _ in range(100):
        X_raw = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d),
                                              size=train_size + 5000)
        _, y_raw, _, _ = label_generator(
            args.problem, X_raw, param=_param, difficulty=difficulty, important=important_dims)
        clf.fit(X_raw[:train_size], y_raw[:train_size])
        test_acc = clf.score(X_raw[train_size:], y_raw[train_size:])
        if test_acc > target_accuracy:
            break
        _param *= 1.1
    print('Performance using the whole training set = {0:.2f}'.format(test_acc))

    return X_raw, y_raw


def label_generator(problem, X, param, difficulty=1, beta=None, important=None):
    if important is None or important > X.shape[-1]:
        important = X.shape[-1]
    dim_latent = sum([important ** i for i in range(1, difficulty + 1)])
    if beta is None:
        beta = np.random.normal(size=[1, dim_latent])
    important_dims = np.random.choice(X.shape[-1], important, replace=False)
    funct_init = lambda inp: np.sum(beta * generate_features(inp[:, important_dims], difficulty), -1)
    batch_size = max(100, min(len(X), 10000000 // dim_latent))
    y_true = np.zeros(len(X))
    while True:
        try:
            for itr in range(int(np.ceil(len(X) / batch_size))):
                y_true[itr * batch_size: (itr + 1) * batch_size] = funct_init(
                    X[itr * batch_size: (itr + 1) * batch_size])
            break
        except MemoryError:
            batch_size = batch_size // 2
    mean, std = np.mean(y_true), np.std(y_true)
    funct = lambda x: (np.sum(beta * generate_features(
        x[:, important_dims], difficulty), -1) - mean) / std
    y_true = (y_true - mean) / std
    if problem is 'classification':
        y_true = logistic.cdf(param * y_true)
        y = (np.random.random(X.shape[0]) < y_true).astype(int)
    elif problem is 'regression':
        y = y_true + param * np.random.normal(size=len(y_true))
    else:
        raise ValueError('Invalid problem specified!')
    return beta, y, y_true, funct

def generate_features(latent, dependency):

    features = []
    n = latent.shape[0]
    exp = latent
    holder = latent
    for order in range(1,dependency+1):
        features.append(np.reshape(holder,[n,-1]))
        exp = np.expand_dims(exp,-1)
        holder = exp * np.expand_dims(holder,1)
    return np.concatenate(features,axis=-1)


