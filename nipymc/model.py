import pandas as pd
import numpy as np
from .convolutions import get_convolution
from sklearn.preprocessing import scale as standardize
from scipy.signal import detrend as lin_detrend
import pymc3 as pm
from six import string_types
from collections import OrderedDict
from sklearn.linear_model import LinearRegression


class BayesianModel(object):

    def __init__(self, dataset, convolution='spm_hrf', conv_kws=None):
        '''
        Args:
            dataset (Dataset): the Dataset instance to draw data from
            convolution (str): name of the default convolution to use. Must be
                a method name in convolutions.py.
            conv_kws (dict): optional dictionary of keyword arguments to pass
                on to the selected convolution function.
        '''
        self.dataset = dataset
        if conv_kws is None:
            # conv_kws = {'hz': 1./self.dataset.TR}
            conv_kws = {'tr': self.dataset.TR}
        self.convolution = get_convolution(convolution, **conv_kws)
        self._index_events()
        self.reset()

    def reset(self):
        self.X = []
        self.dists = {}
        self.model = pm.Model()
        self.mu = 0.
        self.cache = {}
        self.shared_params = {}

    def _index_events(self):
        ''' Calculate and store the row position of all events relative to the
        concatenated activation data so that they align correctly.'''
        # TODO: rounding to nearest TR is probably fine, but if we want to be
        # careful, we should eventually interpolate properly.
        n_vols, n_runs = self.dataset.n_vols, self.dataset.n_runs
        events = self.dataset.design
        onset_tr = np.floor(events['onset'] / self.dataset.TR).astype(int)
        subjects = self.dataset.activation['subject'].unique()
        subject_map = dict(zip(subjects, list(range(len(subjects)))))
        shift = events['subject'].replace(subject_map) * n_vols * n_runs
        events['onset_row'] = onset_tr + shift
        self.events = events

    def _get_variable_data(self, variable, categorical, trend=None):
        ''' Extract (and cache) design matrix for variable/categorical combo.
        '''

        events = self.events.copy()

        cache_key = hash((variable, categorical))

        if cache_key not in self.cache:

            n_rows = len(self.dataset.activation)

            # Handle special cases
            if variable == 'intercept':
                dm = np.ones((n_rows, 1))
            elif variable in ['subject', 'run']:
                n_vols, n_runs = self.dataset.n_vols, self.dataset.n_runs
                n_grps = self.dataset.activation['subject'].nunique()
                if variable == 'run':
                    n_grps *= n_runs
                else:
                    n_vols *= n_runs
                dm = np.zeros((n_rows, n_grps))
                val = 1 if trend is None else standardize(
                    np.arange(n_vols)**trend)
                for i in range(n_grps):
                    dm[(n_vols*i):(n_vols*i+n_vols), i] = val
            else:
                if variable not in events.columns:
                    raise ValueError("No variable '%s' found in the input "
                                     "dataset!" % variable)

                # Initialize design matrix
                n_cols = events[variable].nunique() if categorical else 1
                dm = np.zeros((n_rows, n_cols))

                idx = events['onset_row']

                # For categorical variables, map unique values onto numerical indices,
                # and return data as a DataFrame where each column is a (named) level
                # of the variable
                if categorical:
                    levels = events[variable].unique()
                    mapping = OrderedDict(zip(levels, list(range(n_cols))))
                    events[variable] = events[variable].replace(mapping)
                    dm[idx, events[variable]] = 1.

                # For continuous variables, just index into array
                else:
                    dm[idx, 0] = events[variable]

                # Convolve with boxcar to account for event duration
                duration_tr = np.round(
                    events['duration'] / self.dataset.TR).astype(int)
                # TODO: allow variable duration across events. Can do this trivially
                # by looping over events; can non-uniform convolution be
                # vectorized?
                if len(np.unique(duration_tr)) > 1:
                    raise ValueError("At the moment, all events must have identical "
                                     "duration.")
                filt = np.ones(duration_tr.iloc[0])
                dm = self._convolve(dm, filt)

            self.cache[cache_key] = dm

        dm = self.cache[cache_key]

        return dm

    def _get_membership_graph(self, target_var, group_var):
        ''' Return a T x G binary matrix, where T and G are the number of
        unique values in the target_var and group_var, respectively. Values of
        1 indicate a group membership relation (e.g., stimuli in rows can
        belong (1) or not belong (0) to categories in columns. Only applies to
        categorical variables.
        '''
        targ_levels = self.events[target_var].unique()
        grp_levels = self.events[group_var].unique()
        ct = pd.crosstab(self.events[target_var], self.events[group_var])
        ct[ct > 1] = 1  # binarize
        return ct.loc[targ_levels, grp_levels]  # make sure order is correct

    def _orthogonalize(self, target, covariates):
        ''' Residualize each column in target on all columns in all covariates.
        Args:
            target (ndarray): 2D array of timepoints x predictors
            covariates (list): list of names of categorical fixed effects
                variables to residualize the target on.
            orthogonalize the target with respect to, or a list of ndarrays.
        Returns:
            ndarray with same dimensions as target.
        '''
        # TODO: this can probably all be done in one or two matrix operations
        # instead of looping over target columns and covariates.
        resids = np.zeros_like(target)
        est = LinearRegression()
        if not isinstance(covariates, list):
            covariates = [covariates]
        covariates = [self._get_variable_data(c, True) for c in covariates]
        for i in range(target.shape[1]):
            col_resids = target[:, i]
            for cov in covariates:
                est.fit(cov, col_resids)
                col_resids = target[:, i] - est.predict(cov)
            resids[:, i] = col_resids
        return resids

    @staticmethod
    def _convolve(arr, filter):
        ''' Wrapper for numpy convolution function. Need to convolve in
        'full' mode and then truncate, otherwise we get nastly circular
        convolution effects at the boundary for some convolution functions
        '''
        trim = np.apply_along_axis(
            lambda x: np.convolve(x, filter, mode='full'),
            axis=0, arr=arr)[:len(arr), :]
        return trim

    def _build_dist(self, label, dist, overwrite=False, **kwargs):
        ''' Build a PyMC3 Distribution and store it in the self.dists dict.
        '''
        if overwrite or label not in self.dists:
            if isinstance(dist, string_types):
                if not hasattr(pm, dist):
                    raise ValueError("The Distribution class '%s' was not "
                                     "found in PyMC3." % dist)
                dist = getattr(pm, dist)
            self.dists[label] = dist(label, **kwargs)
        return self.dists[label]

    def add_term(self, variable, label=None, categorical=False, random=False,
                 split_by=None, yoke_random_mean=False, dist='Normal',
                 scale=None, trend=None, orthogonalize=None, convolution=None,
                 conv_kws=None, sigma_kws=None, withhold=False, plot=False,
                 **kwargs):
        '''
        Args:
            variable (str): name of the variable in the Dataset that contains
                the predictor data for the term
            label (str): short name/label of the term; will be used as the
                name passed to PyMC. If None, the variable name is used.
            categorical (bool): if False, treat the input data as continuous;
                if True, treats input as categorical, and assigns discrete
                levels to different columns in the predictor matrix
            random (bool): if False, model as fixed effect; if True, model as
                random effect
            split_by (str): optional name of another variable on which to split
                the target variable. A separate hyperparameter will be included
                for each level in the split_by variable. E.g., if variable = 
                'stimulus' and split_by = 'category', the model will include
                one parameter for each individual stimulus, plus C additional
                hyperparameters for the stimulus variances (one per category).
            yoke_random_mean (bool): 
            dist (str, Distribution): the PyMC3 distribution to use for the
                prior. Can be either a string (must be the name of a class in
                pymc3.distributions), or an uninitialized Distribution object.
            scale (str, bool): if 'before', scaling will be applied before
                convolving with the HRF. If 'after', scaling will be applied to
                the convolved regressor. True is treated like 'before'. If
                None (default), no scaling is done.
            trend (int): if variable is 'subject' or 'run', passing an int here
                will result in addition of an Nth-order polynomial trend
                instead of the expected intercept. E.g., when variable = 
                'run' and trend = 1, a linear trend will be added for each run.
            orthogonalize (list): list of variables to orthogonalize the target
                variable with respect to. For now, this only works for
                categorical covariates. E.g., if variable = 'condition' and
                orthogonalize = ['stimulus_category'], each level of condition
                will be residualized on all (binarized) levels of stimulus
                condition.
            convolution (str): the name of the convolution function to apply
                to the input data; must be a valid function in convolutions.py.
            conv_kws (dict): optional dictionary of additional keyword
                arguments to pass onto the selected convolution function.
            sigma_kws (dict): optional dictionary of keyword arguments
                specifying the parameters of the Distribution to use as the
                sigma for a random variable. Defaults to HalfCauchy with
                beta=10. Ignored unless random=True.
            withhold (bool): if True, the PyMC distribution(s) will be created
                but not added to the prediction equation. This is useful when,
                e.g., yoking the mean of one distribution to the estimated
                value of another distribution, without including the same
                quantity twice.
            plot (bool): if True, plots the resulting design matrix component.
            kwargs: optional keyword arguments passed onto the selected PyMC3
                Distribution.
        '''

        # Load design matrix for requested variable
        dm = self._get_variable_data(variable, categorical, trend=trend)

        if label is None:
            label = variable

        n_cols = dm.shape[1]

        # Handle random effects with nesting/crossing
        if split_by is not None:
            split_dm = self._get_variable_data(split_by, True)
            dm = np.einsum('ab,ac->abc', dm, split_dm)

        # Orthogonalization
        # TODO: generalize this to handle any combination of settings; right
        # now it will only work properly when both the target variable and the
        # covariates are categorical fixed effects.
        if orthogonalize is not None:
            dm = self._orthogonalize(dm, orthogonalize)

        # Scaling and HRF: apply over last dimension
        if dm.ndim == 2:
            dm = dm[..., None]

        for i in range(dm.shape[-1]):

            if scale and scale != 'after':
                dm[..., i] = standardize(dm[..., i])

            # Convolve with HRF
            if variable not in ['subject', 'run', 'intercept']:
                if convolution is None:
                    convolution = self.convolution
                elif not hasattr(convolution, 'shape'):
                    convolution = get_convolution(convolution, conv_kws)

                _convolved = self._convolve(dm[..., i], convolution)
                dm[..., i] = _convolved  # np.squeeze(_convolved)

            if scale == 'after':
                dm[..., i] = standardize(dm[..., i])

            if plot:
                self.plot_design_matrix(dm[..., i])

        if dm.shape[-1] == 1:
            dm = dm.reshape(dm.shape[:2])

        with self.model:

            # Random effects
            if random:

                # User can pass sigma specification in sigma_kws.
                # If not provided, default to HalfCauchy with beta = 10.
                if sigma_kws is None:
                    sigma_kws = {'dist': 'HalfCauchy', 'beta': 10}

                if split_by is None:
                    sigma = self._build_dist('sigma_' + label, **sigma_kws)
                    u = self._build_dist('u_' + label, dist, mu=0., sd=sigma,
                                         shape=n_cols, **kwargs)
                    self.mu += pm.dot(dm, u)
                else:
                    id_map = self._get_membership_graph(variable, split_by)
                    for i in range(id_map.shape[1]):
                        group_items = id_map.iloc[:, i].astype(bool).values
                        selected = dm[:, group_items, i]
                        name = '%s_%s' % (label, id_map.columns[i])
                        sigma = self._build_dist('sigma_' + name, **sigma_kws)
                        if yoke_random_mean:
                            mu = self.dists['b_' + split_by][i]
                        else:
                            mu = 0.
                        u = self._build_dist('u_' + name, dist, mu=mu, sd=sigma,
                                             shape=selected.shape[1], **kwargs)
                        self.mu += pm.dot(selected, u)
            # Fixed effects
            else:
                b = self._build_dist('b_' + label, dist, shape=dm.shape[-1],
                                         **kwargs)
                if split_by is not None:
                    dm = np.squeeze(dm)
                if not withhold:
                    self.mu += pm.dot(dm, b)

        return dm

    def add_deterministic(self, label, expr):
        ''' Add a deterministic variable by evaling the passed expression. '''
        from theano import tensor as T
        with self.model:
            self._build_dist(label, 'Deterministic', var=eval(expr))

    def _setup_y(self, y_data, ar):
        from theano import shared
        from theano import tensor as T
        ''' Sets up y to be a theano shared variable. '''
        if 'y' not in self.shared_params:
            self.shared_params['y'] = shared(y_data)
            with self.model:
                sigma = pm.HalfCauchy('sigma_y_obs', beta=10)
                for i in range(1, ar+1):
                    smoothing_param = pm.Cauchy('AR(%d)' % i, alpha=0, beta=1)
                    _dummy = shared(np.zeros((i,)))
                    _trunc = self.shared_params['y'][:-i]
                    _ar = T.concatenate((_dummy, _trunc)) * smoothing_param
                    self.mu += _ar
                y_obs = pm.Normal('Y_obs', mu=self.mu, sd=sigma, 
                                  observed=self.shared_params['y'])
        else:
            self.shared_params['y'].set_value(y_data)

    def set_y(self, y, detrend=True, scale=None, ar=0):
        ''' Set the outcome variable.
        Args:
            name (str, int): the name (if str) or index (if int) of the
                variable to set as the outcome.
            scale (str): whether and how to standardize the data. If None,
                no scaling is applied. If 'subject', each subject's data are
                scaled independently; if 'run', each run's data are scaled
                independently.
            ar (int): the order of autoregressive coefficients to include.
                If 0, no AR term will be added.
        '''

        self.y = y
        if isinstance(y, int):
            y = self.dataset.activation.columns[y]
        if scale is None:
            y_data = self.dataset.activation.loc[:, y].values
        else:
            if scale == 'run':
                scale = ['subject', 'run']
            y_grps = self.dataset.activation.groupby(scale)
            y_data = y_grps[y].transform(standardize).values
        if detrend:
            y_grps = self.dataset.activation.loc[:, ['subject', 'run']]
            y_grps[y] = y_data
            y_grps = y_grps.groupby(['subject', 'run'])
            y_data = y_grps[y].transform(lin_detrend).values

        self._setup_y(y_data, ar)
        return y_data

    def run(self, samples=1000, find_map=True, verbose=False, step='nuts',
            burn=0.5, **kwargs):
        ''' Run the model.
        Args:
            samples (int): Number of MCMC samples to generate
            find_map (bool): passed to find_map argument of pm.sample()
            verbose (bool): if True, prints additional informatino
            step (str or PyMC3 Sampler): either an instantiated PyMC3 sampler,
                or the name of the sampler to use (either 'nuts' or
                'metropolis').
            burn (int or float): Number or proportion of samples to treat as
                burn-in; passed onto the BayesianModelResults instance returned
                by this method.
            kwargs (dict): optional keyword arguments passed on to the sampler.

        Returns: an instance of class BayesianModelResults.

        '''
        with self.model:
            njobs = kwargs.pop('njobs', 1)
            if isinstance(step, string_types):
                step = {
                    'nuts': pm.NUTS,
                    'metropolis': pm.Metropolis
                }[step.lower()](**kwargs)

            start = kwargs.get('start',
                               pm.find_MAP() if find_map else None)
            self.start = start
            trace = pm.sample(
                samples, start=start, step=step, progressbar=verbose, njobs=njobs)
            self.last_trace = trace  # for convenience
            return BayesianModelResults(trace)

    def add_intercept(self, level='subject'):
        pass

    def add_trend(self, level='run', order=1):
        pass

    def plot_design_matrix(self, dm, panel=False):
        import matplotlib.pyplot as plt
        import seaborn as sns
        n_cols = dm.shape[1]
        plt.figure(figsize=(0.005*len(dm), 0.5*n_cols))
        colors = sns.husl_palette(n_cols)
        for i in range(n_cols):
            plt.plot(dm[:, i], c=colors[i], lw=2)
        plt.show()


class BayesianModelResults(object):

    def __init__(self, trace, burn=0.5):
        self.trace = trace
        if isinstance(burn, float):
            n = trace.varnames[0]
            burn = round(len(trace[n]) * burn)
        self.burn = burn

    def summarize(self, terms=None, contrasts=None):
        if terms is None:
            terms = self.trace.varnames
        summary = {'terms': {}, 'contrasts': {}}
        len_trace = len(self.trace[terms[0], self.burn:])
        summary['n_samples'] = len_trace
        for t in terms:
            _tr = self.trace[t, self.burn:]
            n_cols = 1 if len(_tr.shape) == 1 else _tr.shape[1]
            mu = _tr.mean(0)
            sd = _tr.std(0)
            z = mu/sd
            summary['terms'][t] = dict(zip(['m', 'sd', 'z'], [mu, sd, z]))
            # For now, contrast every pair of levels of all the fixed effects,
            # and each level vs. all others. Extend this to take custom
            # contrasts later.
            if t.startswith('b_'):
                levels = list(range(n_cols))
                for i in range(n_cols-1):
                    for j in range(1, n_cols):
                        if i == j: continue
                        c_name = '%s_%d_vs_%d' % (t, i, j)
                        summary['contrasts'][c_name] = self.compare(t, i, j)
                    c_name = '%s_%d_vs_all' % (t, i)
                    _levs = list(levels)
                    _levs.remove(i)
                    summary['contrasts'][c_name] = self.compare(t, i, _levs)
        return summary

    def compare(self, name, l1=0, l2=1):
        samp1 = self.trace[name][self.burn:, l1]
        if len(samp1.shape) > 1:
            samp1 = samp1.mean(1)

        if l2 is None:
            mu = samp1
        else:
            samp2 = self.trace[name][self.burn:, l2]
            if len(samp2.shape) > 1:
                samp2 = samp2.mean(1)
            mu = samp1 - samp2
            mu_m = mu.mean()
            mu_sd = mu.std()

        return(mu_m, mu_sd,  mu_m/mu_sd)

    def save(self, filename):
        pass

    @classmethod
    def load(cls):
        pass
