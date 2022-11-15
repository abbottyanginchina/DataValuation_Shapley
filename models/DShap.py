import numpy as np
import os
import warnings

import torch.random
import _pickle as pkl
import inspect
from sklearn.base import clone
from sklearn.metrics import f1_score
from utils.shap_utils import my_auc_score, my_xe_score

from models.Nets import return_model
from utils.parameters import args_parser
from utils.save import delete_file

args = args_parser()

class DShap(object):

    '''
    Args:
        X: Data covariates
        y: Data labels
        X_test: Test+Held-out covariates
        y_test: Test+Held-out labels
    '''

    def __init__(self, X, y, X_test, y_test, num_test, sources=None,
                 sample_weight=None, directory=None, problem='classification',
                 model_family='logistic', metric='accuracy', seed=None,
                 overwrite=False,
                 **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            samples_weights: Weight of train samples in the loss function
                (for models where weighted training method is enabled.)
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting
                same permutations.
            overwrite: Delete existing data and start computations from
                scratch
            **kwargs: Arguments of the model
        """

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)  # 固定随机种子（CPU）
            if torch.cuda.is_available():  # 固定随机种子（GPU)
                torch.cuda.manual_seed(seed)  # 为当前GPU设置
                torch.cuda.manual_seed_all(seed)  # 为所有GPU设置

        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        if self.model_family is 'logistic':
            self.hidden_units = []
        if self.directory is not None:
            if overwrite and os.path.exists(directory):
                delete_file(directory)

            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test,
                                      sources, sample_weight)
        if len(set(self.y)) > 2:
            assert self.metric != 'f1', 'Invalid metric for multiclass!'
            assert self.metric != 'auc', 'Invalid metric for multiclass!'
        is_regression = (np.mean(self.y//1 == self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        if self.is_regression:
            warnings.warn("Regression problem is no implemented.")
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)

    def _initialize_instance(self, X, y, X_test, y_test, num_test,
                             sources=None, sample_weight=None):
        """Loads or creates sets of data."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        data_dir = os.path.join(self.directory, 'data.pkl')
        if os.path.exists(data_dir):
            self._load_dataset(data_dir)
        else:
            self.X_heldout = X_test[:-num_test]
            self.y_heldout = y_test[:-num_test]
            self.X_test = X_test[-num_test:]
            self.y_test = y_test[-num_test:]
            self.X, self.y, self.sources = X, y, sources
            self.sample_weight = sample_weight
            data_dic = {'X': self.X, 'y': self.y, 'X_test': self.X_test,
                        'y_test': self.y_test, 'X_heldout': self.X_heldout,
                        'y_heldout': self.y_heldout, 'sources': self.sources}
            if sample_weight is not None:
                data_dic['sample_weight'] = sample_weight
                warnings.warn("Sample weight not implemented for G-Shapley")
            pkl.dump(data_dic, open(data_dir, 'wb'))
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = pkl.load(open(loo_dir, 'rb'))['loo']
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        n_points = len(self.X)

        self.tmc_number, self.g_number = self._which_parallel(self.directory)
        self._create_results_placeholder(
            self.directory, self.tmc_number, self.g_number,
            n_points, n_sources, self.model_family)

    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                      for name in previous_results if 'mem_tmc' in name]
        g_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_g' in name]
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0'
        g_number = str(np.max(g_nmbrs) + 1) if len(g_nmbrs) else '0'
        return tmc_number, g_number

    def _create_results_placeholder(self, directory, tmc_number, g_number,
                                   n_points, n_sources, model_family):
        tmc_dir = os.path.join(
            directory,
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            directory,
            'mem_g_{}.pkl'.format(g_number.zfill(4))
        )
        self.mem_tmc = np.zeros((0, n_points))
        self.mem_g = np.zeros((0, n_points))
        self.idxs_tmc = np.zeros((0, n_sources), int)
        self.idxs_g = np.zeros((0, n_sources), int)
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
                 open(tmc_dir, 'wb'))
        if model_family not in ['logistic', 'NN']:
            return
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g},
                 open(g_dir, 'wb'))

    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float) / len(self.y_test)
            return np.max(hist)
        if metric == 'f1':
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(self.y_test)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            rnd_y = np.random.permutation(self.y)
            if self.sample_weight is None:
                self.model.fit(self.X, rnd_y)
            else:
                self.model.fit(self.X, rnd_y,
                               sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)

    def restart_model(self):

        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)

    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data
                different from test set.
            y: Labels, if valuation is performed on a data
                different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if inspect.isfunction(metric):
            return metric(model, X, y)
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')

    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(
            self.directory,
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            self.directory,
            'mem_g_{}.pkl'.format(self.g_number.zfill(4))
        )
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc},
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g},
                 open(g_dir, 'wb'))

    def _calculate_loo_vals(self, sources=None, metric=None):
        """Calculated leave-one-out values for the given metric.

        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.

        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}

        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric
        self.restart_model()
        if self.sample_weight is None:
            self.model.fit(self.X, self.y)
        else:
            self.model.fit(self.X, self.y,
                           sample_weight=self.sample_weight)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(len(self.X))

        for i in sources.keys():
            X_batch = np.delete(self.X, sources[i], axis=0)
            y_batch = np.delete(self.y, sources[i], axis=0)
            if self.sample_weight is not None:
                sw_batch = np.delete(self.sample_weight, sources[i], axis=0)
            if self.sample_weight is None:
                self.model.fit(X_batch, y_batch)
            else:
                self.model.fit(X_batch, y_batch, sample_weight=sw_batch)

            removed_value = self.value(self.model, metric=metric)
            vals_loo[sources[i]] = (baseline_value - removed_value)
            vals_loo[sources[i]] /= len(sources[i])

        return vals_loo

    def run(self, save_every, err, tolerance=0.01, g_run=True, loo_run=True):
        """
        Calculates data sources(points) values.

        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
        if loo_run:
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
                self.save_results(overwrite=True)
            print('LOO values calculated!')



