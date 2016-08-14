import pandas as pd
import numpy as np
from .convolutions import get_convolution
from sklearn.preprocessing import scale as standardize
from scipy.signal import detrend as lin_detrend
import pymc3 as pm
from six import string_types
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from .utils import listify


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
        self.events = self.dataset.design.copy().reset_index()
        self.reset()

    def reset(self):
        self.X = []
        self.dists = {}
        self.model = pm.Model()
        self.mu = 0.
        self.cache = {}
        self.shared_params = {}
        self.level_map = {}

    def _get_variable_data(self, variable, categorical, label=None, trend=None):
        ''' Extract (and cache) design matrix for variable/categorical combo.
        '''

        # assign default labels (for variables passed in via split_by or orthogonalize)
        if label is None:
            label = '_'.join(listify(variable))

        # hash labels (rather than variables)
        cache_key = hash((label, categorical))

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
                run_dms = []
                events = self.events.copy()
                sr = 100  # Sampling rate, in Hz
                events['run_onset'] = (events['run_onset'] * sr).round()
                events['duration'] = (events['duration'] * sr).round()
                tr = self.dataset.TR
                scale = np.ceil(tr * sr)
                n_rows = self.dataset.n_vols * scale

                if categorical:
                    variable_cols = events[variable]
                    if isinstance(variable, (list, tuple)):
                        variable_cols = variable_cols.stack()
                    n_cols = variable_cols.nunique()

                    # map unique values onto numerical indices, and return
                    # data as a DataFrame where each column is a (named) level
                    # of the variable
                    levels = variable_cols.unique()
                    mapping = OrderedDict(zip(levels, list(range(n_cols))))
                    if label is not None:
                        self.level_map[label] = mapping
                    events[variable] = events[variable].replace(mapping)

                else:
                    n_cols = 1

                for (sub_, run_), g in events.groupby(['subject', 'run']):
                    dm = np.zeros((n_rows, n_cols))
                    for i, row in g.iterrows():
                        start = int(row['run_onset'])
                        end = int(start + row['duration'])

                        if categorical:
                            for var in listify(variable):
                                dm[start:end, row[variable]] = 1
                        else:
                            if isinstance(variable, (tuple, list)):
                                raise ValueError("Adding a list of terms is only "
                                        "supported for categorical variables "
                                        "(e.g., random factors).")
                            dm[start:end, 0] = row[variable]

                    dm = dm.reshape(-1, scale, n_cols).mean(axis=1)
                    run_dms.append(dm[:self.dataset.n_vols])

                dm = np.concatenate(run_dms)

            self.cache[cache_key] = dm

        dm = self.cache[cache_key]

        # NOTE: we return a copy in order to avoid in-place changes to the
        # cached design matrix (e.g., we don't want the HRF convolution to
        # overwrite what's in the cache).
        return dm.copy()

    def _get_membership_graph(self, target_var, group_var):
        ''' Return a T x G binary matrix, where T and G are the number of
        unique values in the target_var and group_var, respectively. Values of
        1 indicate a group membership relation (e.g., stimuli in rows can
        belong (1) or not belong (0) to categories in columns. Only applies to
        categorical variables.
        '''
        if isinstance(target_var, (list, tuple)):
            targ_levels = self.events[target_var].stack().unique()
        else:
            targ_levels = self.events[target_var].unique()
        grp_levels = self.events[group_var].unique()
        if isinstance(target_var, (list, tuple)):
            ct = pd.crosstab(self.events[target_var].stack().values,
                             self.events[group_var].repeat(len(target_var)).values)
        else:
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
                 split_by=None, yoke_random_mean=False, estimate_random_mean=False,
                 dist='Normal', scale=None, trend=None, orthogonalize=None,
                 convolution=None, conv_kws=None, sigma_kws=None, withhold=False,
                 plot=False, **kwargs):
        '''
        Args:
            variable (str): name of the variable in the Dataset that contains
                the predictor data for the term, or a list of variable names.
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
            estimate_random_mean (bool): If False (default), set mean of random
                effect distribution to 0. If True, estimate mean parameters for 
                each level of split_by (in which case the corresponding fixed
                effect parameter should be omitted, for identifiability reasons).
                If split_by=None, this is equivalent to estimating a fixed
                intercept term. Note that models parameterized in this way are
                often less numerically stable than the default parameterization.
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
                If None, the default convolution function set at class
                initialization is used. If 'none' is passed, no convolution
                at all is applied.
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

        if label is None:
            label = '_'.join(listify(variable))

        # Load design matrix for requested variable
        dm = self._get_variable_data(variable, categorical, label=label,
                                     trend=trend)
        n_cols = dm.shape[1]

        # Handle random effects with nesting/crossing. Basically this splits the design
        # matrix into a separate matrix for each level of split_by, stacked into 3D array
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
        # if there is no split_by, add a dummy 3rd dimension so code below works in general
        if dm.ndim == 2:
            dm = dm[..., None]

        for i in range(dm.shape[-1]):

            if scale and scale != 'after':
                dm[..., i] = standardize(dm[..., i])

        if plot:
            self.plot_design_matrix(dm, variable, split_by)

            # Convolve with HRF
            if variable not in ['subject', 'run', 'intercept'] and convolution is not 'none':
                if convolution is None:
                    convolution = self.convolution
                elif not hasattr(convolution, 'shape'):
                    convolution = get_convolution(convolution, conv_kws)

                _convolved = self._convolve(dm[..., i], convolution)
                dm[..., i] = _convolved  # np.squeeze(_convolved)

            if scale == 'after':
                dm[..., i] = standardize(dm[..., i])

        # remove the dummy 3rd dimension if it was added prior to scaling/convolution
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
                    if estimate_random_mean:
                        mu = self._build_dist('b_' + label, dist)
                    else:
                        mu = 0.
                    u = self._build_dist('u_' + label, dist, mu=mu, sd=sigma,
                                         shape=n_cols, **kwargs)
                    self.mu += pm.dot(dm, u)
                else:
                    # id_map is essentially a crosstab except each cell is either 0 or 1
                    id_map = self._get_membership_graph(variable, split_by)
                    for i in range(id_map.shape[1]):
                        # select just the factor levels that appear with the
                        # current level of split_by
                        group_items = id_map.iloc[:, i].astype(bool)
                        selected = dm[:, group_items.values, i]
                        # add the level effects to the model
                        name = '%s_%s' % (label, id_map.columns[i])
                        sigma = self._build_dist('sigma_' + name, **sigma_kws)
                        if yoke_random_mean:
                            mu = self.dists['b_' + split_by][i]
                        elif estimate_random_mean:
                            mu = self._build_dist('b_' + name, dist)
                        else:
                            mu = 0.
                        name, size = 'u_' + name, selected.shape[1]
                        u = self._build_dist(name, dist, mu=mu, sd=sigma,
                                             shape=size, **kwargs)
                        self.mu += pm.dot(selected, u)

                        # Update the level map
                        levels = group_items[group_items].index.tolist()
                        self.level_map[name] = OrderedDict(zip(levels, list(range(size))))

            # Fixed effects
            else:
                b = self._build_dist('b_' + label, dist, shape=dm.shape[-1],
                                         **kwargs)
                if split_by is not None:
                    dm = np.squeeze(dm)
                if not withhold:
                    self.mu += pm.dot(dm, b)

        # return dm

    def add_deterministic(self, label, expr):
        ''' Add a deterministic variable by evaling the passed expression. '''
        from theano import tensor as T
        with self.model:
            self._build_dist(label, 'Deterministic', var=eval(expr))

    def _setup_y(self, y_data, ar, by_run):
        from theano import shared
        from theano import tensor as T
        ''' Sets up y to be a theano shared variable. '''
        if 'y' not in self.shared_params:
            self.shared_params['y'] = shared(y_data)

            with self.model:

                n_vols = self.dataset.n_vols
                n_runs = int(len(y_data) / n_vols)

                for i in range(1, ar+1):

                    _pad = shared(np.zeros((i,)))
                    _trunc = self.shared_params['y'][:-i]
                    y_shifted = T.concatenate((_pad, _trunc))
                    weights = np.r_[np.zeros(i), np.ones(n_vols-i)]

                    # Model an AR term for each run or use just one for all runs
                    if by_run:
                        smoother = pm.Cauchy('AR(%d)' % i, alpha=0, beta=1, shape=n_runs)
                        weights = np.outer(weights, np.eye(n_runs))
                        weights = np.reshape(weights, (n_vols*n_runs, n_runs), order='F')
                        _ar = pm.dot(weights, smoother) * y_shifted
                    else:
                        smoother = pm.Cauchy('AR(%d)' % i, alpha=0, beta=1)
                        weights = np.tile(weights, n_runs)
                        _ar = shared(weights) * y_shifted * smoother

                    self.mu += _ar

                sigma = pm.HalfCauchy('sigma_y_obs', beta=10)
                y_obs = pm.Normal('Y_obs', mu=self.mu, sd=sigma, 
                                  observed=self.shared_params['y'])
        else:
            self.shared_params['y'].set_value(y_data)

    def set_y(self, y, detrend=True, scale=None, ar=0, by_run=False):
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
            by_run (bool): whether to include a separate set of AR terms for
                each scanning run (True), or use the same set of terms for all
                runs (False; default). The former is the technically correct
                approach, but is computationally expensive, and in practice,
                usually has a negligible impact on results.
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

        self._setup_y(y_data, ar, by_run)

    def run(self, samples=1000, find_map=True, verbose=True, step='nuts',
            burn=0.5, **kwargs):
        ''' Run the model.
        Args:
            samples (int): Number of MCMC samples to generate
            find_map (bool): passed to find_map argument of pm.sample()
            verbose (bool): if True, prints additional information
            step (str or PyMC3 Sampler): either an instantiated PyMC3 sampler,
                or the name of the sampler to use (either 'nuts' or
                'metropolis').
            start: Optional starting point to pass onto sampler.
            burn (int or float): Number or proportion of samples to treat as
                burn-in; passed onto the BayesianModelResults instance returned
                by this method.
            kwargs (dict): optional keyword arguments passed on to the sampler.

        Returns: an instance of class BayesianModelResults.

        '''
        with self.model:
            njobs = kwargs.pop('njobs', 1)
            start = kwargs.pop('start', pm.find_MAP() if find_map else None)
            chain = kwargs.pop('chain', 0)
            if isinstance(step, string_types):
                step = {
                    'nuts': pm.NUTS,
                    'metropolis': pm.Metropolis
                }[step.lower()](**kwargs)

            self.start = start
            trace = pm.sample(
                samples, start=start, step=step, progressbar=verbose, njobs=njobs, chain=chain)
            self.last_trace = trace  # for convenience
            return BayesianModelResults(trace)

    def add_intercept(self, level='subject'):
        pass

    def add_trend(self, level='run', order=1):
        pass

    def plot_design_matrix(self, dm, variable, split_by=None, panel=False):
        print(self.level_map.keys())
        import matplotlib.pyplot as plt
        import seaborn as sns
        n_cols = min(dm.shape[1], 10)
        n_axes = dm.shape[2]
        n_rows = self.dataset.n_vols * 3 * self.dataset.n_runs
        fig, axes = plt.subplots(n_axes, 1, figsize=(20, 2 * n_axes))
        if n_axes == 1:
            axes = [axes]
        colors = sns.husl_palette(n_cols)
        for j in range(n_axes):
            ax = axes[j]
            min_y, max_y = dm[:n_rows, :, j].min(), dm[:n_rows, :, j].max()
            for i in range(n_cols):
                ax.plot(dm[:n_rows, i, j], c=colors[i], lw=2)
            ax.set_ylim(min_y - 0.05*min_y*np.sign(min_y), max_y + 0.05*max_y*np.sign(max_y))
            if n_axes > 1:
                try:
                    ax.set(ylabel=list(self.level_map[split_by].keys())[j])
                except: pass
        title = variable + ('' if split_by is None else ' split by ' + split_by)
        axes[0].set_title(title, fontsize=16)
        plt.show()


class BayesianModelResults(object):

    def __init__(self, trace):
        self.trace = trace

    def summarize(self, terms=None, contrasts=None, burn=0.5):
        # TODO: add a 'thin' parameter with argument k (default = 1) so that
        # every kth sample is saved, the rest are discarded before summarizing

        # handle burn whether it is given as a proportion or a number of samples
        if isinstance(burn, float):
            n = self.trace.varnames[0]
            burn = round(len(self.trace[n]) * burn)
        self.burn = burn

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
