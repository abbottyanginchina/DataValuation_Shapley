import numpy as np

def calculate_loo_vals(self, sources=None, metric=None):
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
