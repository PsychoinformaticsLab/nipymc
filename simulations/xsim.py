import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import pymc3 as pm
from pymc3 import Deterministic
import theano.tensor as T
from sklearn.preprocessing import scale as standardize
import sys, pickle
from random import shuffle
import nipymc
from nipymc import *

# 1st argument = instance of simulation
# 2nd argument = p = number of participants
# 3rd argument = q = number of stimuli
# 4th argument = s = SD of stimulus effects
instance = sys.argv[1]
p = sys.argv[2]
q = sys.argv[3]
s = sys.argv[4]

# global variables...
SAMPLES = 1000
BURN = 100

# double-gamma HRF (https://github.com/poldrack/pybetaseries/blob/master/pybetaseries.py)
def spm_hrf(TR,p=[6,16,1,1,6,0,32]):
    """ An implementation of spm_hrf.m from the SPM distribution
    
    Arguments:
    
    Required:
    TR: repetition time at which to generate the HRF (in seconds)
    
    Optional:
    p: list with parameters of the two gamma functions:
                                                         defaults
                                                        (seconds)
       p[0] - delay of response (relative to onset)         6
       p[1] - delay of undershoot (relative to onset)      16
       p[2] - dispersion of response                        1
       p[3] - dispersion of undershoot                      1
       p[4] - ratio of response to undershoot               6
       p[5] - onset (seconds)                               0
       p[6] - length of kernel (seconds)                   32
    """

    p=[float(x) for x in p]

    fMRI_T = 16.0

    TR=float(TR)
    dt  = TR/fMRI_T
    u   = np.arange(p[6]/dt + 1) - p[5]/dt
    hrf=sp.stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - sp.stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts=np.array(range(np.int(p[6]/TR)))*fMRI_T
    hrf=hrf[list(good_pts)]
    # hrf = hrf([0:(p(7)/RT)]*fMRI_T + 1);
    hrf = hrf/np.sum(hrf);
    return hrf

# function to insert ISIs into a trial list
def insert_ISI(trials, ISI):
    return np.insert(trials, np.repeat(range(1,len(trials)), ISI), 0)

# function to build activation sequence from stimulus list
# because of how ISI is added, length of stimulus list must be a multiple of 4
# output a tidy DataFrame including
# subject info, convolved & unconvolved regressors, random effects, etc.
def build_seq(sub_num, stims, sub_A_sd, sub_B_sd):
    # shuffle stimulus list
    stims = stims.reindex(np.random.permutation(stims.index))
    
    # inter-stimulus interval is randomly selected from [1,2,3,4]
    # the first ISI is removed (so sequence begins with a stim presentation)
    ISI = np.delete(np.repeat([1,2,3,4], len(stims.index)/4, axis=0), 0)
    np.random.shuffle(ISI)
    
    # create matrix of stimulus predictors and add ISIs
    X = np.diag(stims['effect'])
    X = np.apply_along_axis(func1d=insert_ISI, axis=0, arr=X, ISI=ISI)
    
    # reorder the columns so they are in the same order (0-39) for everyone
    X = X[:,[list(stims['stim']).index([i]) for i in range(len(stims.index))]]
    
    # now convolve all predictors with double gamma HRF
    X = np.apply_along_axis(func1d=np.convolve, axis=0, arr=X, v=spm_hrf(1))
    
    # build and return this subject's dataframe
    df = pd.DataFrame(X)
    df['time'] = range(len(df.index))
    df['sub_num'] = sub_num
    # df['sub_intercept'] = np.asscalar(np.random.normal(size=1))
    df['sub_A'] = np.asscalar(np.random.normal(size=1, scale=sub_A_sd))
    df['sub_B'] = np.asscalar(np.random.normal(size=1, scale=sub_B_sd))
    return df

def build_seq_block(sub_num, stims, sub_A_sd, sub_B_sd, block_size):
    # block stimulus list and shuffle within each block
    q = len(stims.index)
    stims = [stims.iloc[:q//2,], stims.iloc[q//2:,]]
    stims = [x.reindex(np.random.permutation(x.index)) for x in stims]
    shuffle(stims)
    stims = [[x.iloc[k:(k+block_size),] for k in range(0, q//2, block_size)] for x in stims]
    stims = pd.concat([val for pair in zip(stims[0], stims[1]) for val in pair])

    # inter-stimulus interval is randomly selected from [1,2,3,4]
    # the first ISI is removed (so sequence begins with a stim presentation)
    ISI = np.delete(np.repeat(2, len(stims.index), axis=0), 0)

    # create matrix of stimulus predictors and add ISIs
    X = np.diag(stims['effect'])
    X = np.apply_along_axis(func1d=insert_ISI, axis=0, arr=X, ISI=ISI)

    # reorder the columns so they are in the same order (0-39) for everyone
    X = X[:,[list(stims['stim']).index([i]) for i in range(len(stims.index))]]

    # now convolve all predictors with double gamma HRF
    X = np.apply_along_axis(func1d=np.convolve, axis=0, arr=X, v=spm_hrf(1))

    # build and return this subject's dataframe
    df = pd.DataFrame(X)
    df['time'] = range(len(df.index))
    df['sub_num'] = sub_num
    # df['sub_intercept'] = np.asscalar(np.random.normal(size=1))
    df['sub_A'] = np.asscalar(np.random.normal(size=1, scale=sub_A_sd))
    df['sub_B'] = np.asscalar(np.random.normal(size=1, scale=sub_B_sd))
    return df

# generalize the code above into a simulation function
def simulate(num_subs, num_stims, A_mean, B_mean, sub_A_sd, sub_B_sd, stim_A_sd,
    stim_B_sd, resid_sd, ar=None, block_size=None):
    # build stimulus list
    stims = np.random.normal(size=num_stims//2, loc=1, scale=stim_A_sd/A_mean).tolist() + \
            np.random.normal(size=num_stims//2, loc=1, scale=stim_B_sd/B_mean).tolist()
    stims = pd.DataFrame({'stim':range(num_stims),
                          'condition':np.repeat([0,1], num_stims//2),
                          'effect':np.array(stims)})
    
    # now build design matrix from stimulus list
    if block_size is None:
        # build event-related design
        data = pd.concat([build_seq(sub_num=i, stims=stims, sub_A_sd=sub_A_sd, sub_B_sd=sub_B_sd) for i in range(num_subs)])
    else:
        # build blocked design
        data = pd.concat([build_seq_block(sub_num=i, stims=stims, sub_A_sd=sub_A_sd, sub_B_sd=sub_B_sd, block_size=block_size) for i in range(num_subs)])

    # add response variable and difference predictor
    if ar is None:
        # build y WITHOUT AR(2) errors
        data['y'] = (A_mean + data['sub_A'])*data.iloc[:,:(num_stims//2)].sum(axis=1).values + \
                    (B_mean + data['sub_B'])*data.iloc[:,(num_stims//2):num_stims].sum(axis=1).values + \
                    np.random.normal(size=len(data.index), scale=resid_sd)
    else:
        # build y WITH AR(2) errors
        data['y'] = np.empty(len(data.index))
        data['y_t-1'] = np.zeros(len(data.index))
        data['y_t-2'] = np.zeros(len(data.index))
        for t in range(len(pd.unique(data['time']))):
            data.loc[t,'y'] = pd.DataFrame(
                (A_mean + data.loc[t,'sub_A'])*data.loc[t, range(num_stims//2)].sum(axis=1).values + \
                (B_mean + data.loc[t,'sub_B'])*data.loc[t, range(num_stims//2, num_stims)].sum(axis=1).values + \
                np.random.normal(size=len(data.loc[t].index), scale=resid_sd)).values
            if t==1:
                data.loc[t,'y'] = pd.DataFrame(data.loc[t,'y'].values + ar[0]*data.loc[t-1,'y'].values).values
                data.loc[t,'y_t-1'] = pd.DataFrame(data.loc[t-1,'y']).values
            if t>1:
                data.loc[t,'y'] = pd.DataFrame(data.loc[t,'y'].values + ar[0]*data.loc[t-1,'y'].values + ar[1]*data.loc[t-2,'y'].values).values
                data.loc[t,'y_t-1'] = pd.DataFrame(data.loc[t-1,'y']).values
                data.loc[t,'y_t-2'] = pd.DataFrame(data.loc[t-2,'y']).values

    # remove random stimulus effects from regressors before fitting model
    data.iloc[:, :num_stims] = data.iloc[:, :num_stims] / stims['effect'].tolist()

    # build design DataFrame
    # create num_subs * num_stims DataFrame
    # where each cell is when that stim was presented for that sub
    # note that this depends on there being no repeated stimulus presentations
    gb = data.groupby('sub_num')
    pres = pd.DataFrame([[next(i-1 for i, val in enumerate(df.iloc[:,stim]) if abs(val) > .0001)
                          for stim in range(num_stims)] for sub_num, df in gb])
    # build the design DataFrame from pres
    design = pd.concat([pd.DataFrame({'onset':pres.iloc[sub,:].sort_values(),
                                      'run_onset':pres.iloc[sub,:].sort_values(),
                                      'stimulus':pres.iloc[sub,:].sort_values().index,
                                      'subject':sub,
                                      'duration':1,
                                      'amplitude':1,
                                      'run':1,
                                      'index':range(pres.shape[1])})
                        for sub in range(num_subs)])
    design['condition'] = stims['condition'][design['stimulus']]

    # build activation DataFrame
    activation = pd.DataFrame({'y':data['y'].values,
                               'vol':data['time'],
                               'run':1,
                               'subject':data['sub_num']})

    # build Dataset object
    dataset = nipymc.data.Dataset(design=design, activation=activation, TR=1)
    
    ####################################
    ############ FIT MODELS ############
    ####################################
    
    # SPM model
    def get_diff(df):
        X = pd.concat([df.iloc[:,:num_stims//2].sum(axis=1),
                       df.iloc[:,num_stims//2:num_stims].sum(axis=1),
                       df['y_t-1'],
                       df['y_t-2']], axis=1)
        beta = pd.stats.api.ols(y=df['y'], x=X, intercept=False).beta
        return pd.Series(beta[1] - beta[0]).append(beta)
    sub_diffs = data.groupby('sub_num').apply(get_diff)

    # fit model with FIXED stim effects (LS-All model)
    with pm.Model():

        # Fixed effects
        b = pm.Normal('fixstim_b', mu=0, sd=10, shape=num_stims)
        if ar is not None:
            ar1 = pm.Cauchy('fixstim_AR1', alpha=0, beta=1)
            ar2 = pm.Cauchy('fixstim_AR2', alpha=0, beta=1)

        # random x1 & x2 slopes for participants
        sigma_sub_A = pm.HalfCauchy('fixstim_sigma_sub_A', beta=10)
        sigma_sub_B = pm.HalfCauchy('fixstim_sigma_sub_B', beta=10)
        u0 = pm.Normal('u0_sub_A_log', mu=0., sd=sigma_sub_A, shape=data['sub_num'].nunique())
        u1 = pm.Normal('u1_sub_B_log', mu=0., sd=sigma_sub_B, shape=data['sub_num'].nunique())

        # now write the mean model
        mu = u0[data['sub_num'].values]*data.iloc[:,:(num_stims//2)].sum(axis=1).values + \
             u1[data['sub_num'].values]*data.iloc[:,(num_stims//2):num_stims].sum(axis=1).values + \
             pm.dot(data.iloc[:, :num_stims].values, b)
        if ar is not None: mu += ar1*data['y_t-1'].values + ar2*data['y_t-2'].values

        # define the condition contrast
        cond_diff = Deterministic('fixstim_cond_diff', T.mean(b[num_stims//2:]) - T.mean(b[:num_stims//2]))
        
        # model for the observed values
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=pm.HalfCauchy('fixstim_sigma', beta=10),
                          observed=data['y'].values)

        # run the sampler
        step = pm.NUTS()
        print('fitting fixstim model...')
        trace0 = pm.sample(SAMPLES, step=step, progressbar=False)   

    # fit model WITHOUT random stim effects
    with pm.Model():

        # Fixed effects
        b1 = pm.Normal('nostim_b_A', mu=0, sd=10)
        b2 = pm.Normal('nostim_b_B', mu=0, sd=10)
        if ar is not None:
            ar1 = pm.Cauchy('nostim_AR1', alpha=0, beta=1)
            ar2 = pm.Cauchy('nostim_AR2', alpha=0, beta=1)

        # random x1 & x2 slopes for participants
        sigma_sub_A = pm.HalfCauchy('nostim_sigma_sub_A', beta=10)
        sigma_sub_B = pm.HalfCauchy('nostim_sigma_sub_B', beta=10)
        u0 = pm.Normal('u0_sub_A_log', mu=0., sd=sigma_sub_A, shape=data['sub_num'].nunique())
        u1 = pm.Normal('u1_sub_B_log', mu=0., sd=sigma_sub_B, shape=data['sub_num'].nunique())

        # now write the mean model
        mu = (b1 + u0[data['sub_num'].values])*data.iloc[:,:(num_stims//2)].sum(axis=1).values + \
             (b2 + u1[data['sub_num'].values])*data.iloc[:,(num_stims//2):num_stims].sum(axis=1).values
        if ar is not None: mu += ar1*data['y_t-1'].values + ar2*data['y_t-2'].values

        # define the condition contrast
        cond_diff = Deterministic('nostim_cond_diff', b2 - b1)

        # model for the observed values
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=pm.HalfCauchy('nostim_sigma', beta=10),
                          observed=data['y'].values)

        # run the sampler
        step = pm.NUTS()
        print('fitting nostim model...')
        trace1 = pm.sample(SAMPLES, step=step, progressbar=False)
    
    # fit model with separate dists + variances
    with pm.Model():

        # Fixed effects
        b1 = pm.Normal('randstim_b_A', mu=0, sd=10)
        b2 = pm.Normal('randstim_b_B', mu=0, sd=10)
        if ar is not None:
            ar1 = pm.Cauchy('randstim_AR1', alpha=0, beta=1)
            ar2 = pm.Cauchy('randstim_AR2', alpha=0, beta=1)

        # random x1 & x2 slopes for participants
        sigma_sub_A = pm.HalfCauchy('randstim_sigma_sub_A', beta=10)
        sigma_sub_B = pm.HalfCauchy('randstim_sigma_sub_B', beta=10)
        u0 = pm.Normal('u0_sub_A_log', mu=0., sd=sigma_sub_A, shape=data['sub_num'].nunique())
        u1 = pm.Normal('u1_sub_B_log', mu=0., sd=sigma_sub_B, shape=data['sub_num'].nunique())

        # random stim intercepts
        sigma_stim_A = pm.HalfCauchy('randstim_sigma_stim_A', beta=10)
        u2 = pm.Normal('randstim_stim_A', mu=0., sd=sigma_stim_A, shape=num_stims//2)
        sigma_stim_B = pm.HalfCauchy('randstim_sigma_stim_B', beta=10)
        u3 = pm.Normal('randstim_stim_B', mu=0., sd=sigma_stim_B, shape=num_stims//2)

        # now write the mean model
        mu = (b1 + u0[data['sub_num'].values])*data.iloc[:,:(num_stims//2)].sum(axis=1).values + \
             (b2 + u1[data['sub_num'].values])*data.iloc[:,(num_stims//2):num_stims].sum(axis=1).values + \
             pm.dot(data.iloc[:, :num_stims//2].values, u2) + pm.dot(data.iloc[:, (num_stims//2):num_stims].values, u3)
        if ar is not None: mu += ar1*data['y_t-1'].values + ar2*data['y_t-2'].values

        # define the condition contrast
        cond_diff = Deterministic('randstim_cond_diff', b2 - b1)

        # model for the observed values
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=pm.HalfCauchy('randstim_sigma', beta=10),
                          observed=data['y'].values)

        # run the sampler
        step = pm.NUTS()
        print('fitting 2dist2var model...')
        trace2 = pm.sample(SAMPLES, step=step, progressbar=False)
    
    # fit FIX_STIM model using pymcwrap
    mod3 = nipymc.model.BayesianModel(dataset)
    mod3.add_term('subject', label='nipymc_fixstim_subject', split_by='condition', categorical=True, random=True)
    mod3.add_term('stimulus', label='nipymc_fixstim_stimulus', categorical=True)
    mod3.groupA = [mod3.level_map['nipymc_fixstim_stimulus'][i] for i in range(num_stims//2)]
    mod3.groupB = [mod3.level_map['nipymc_fixstim_stimulus'][i] for i in range(num_stims//2, num_stims)]
    mod3.add_deterministic('nipymc_fixstim_cond_diff',
        "T.mean(self.dists['b_nipymc_fixstim_stimulus'][self.groupB]) - T.mean(self.dists['b_nipymc_fixstim_stimulus'][self.groupA])")
    mod3.set_y('y', scale=None, detrend=False, ar=0 if ar is None else 2)
    print('fitting nipymc_fixstim model...')
    mod3_fitted = mod3.run(samples=SAMPLES, verbose=False, find_map=False)

    # fit NO_STIM model using pymcwrap
    mod4 = nipymc.model.BayesianModel(dataset)
    mod4.add_term('condition', label='nipymc_nostim_condition', categorical=True, scale=False)
    mod4.add_term('subject', label='nipymc_nostim_subject', split_by='condition', categorical=True, random=True)
    groupA = str(mod4.level_map['nipymc_nostim_condition'][0])
    groupB = str(mod4.level_map['nipymc_nostim_condition'][1])
    mod4.add_deterministic('nipymc_nostim_cond_diff',
        "self.dists['b_nipymc_nostim_condition']["+groupB+"] - self.dists['b_nipymc_nostim_condition']["+groupA+"]")
    mod4.set_y('y', scale=None, detrend=False, ar=0 if ar is None else 2)
    print('fitting nipymc_nostim model...')
    mod4_fitted = mod4.run(samples=SAMPLES, verbose=False, find_map=False)
    
    # fit 2dist2var model using pymcwrap
    mod5 = nipymc.model.BayesianModel(dataset)
    mod5.add_term('condition', label='nipymc_randstim_condition', categorical=True, scale=False)
    mod5.add_term('stimulus', label='nipymc_randstim_stimulus', split_by='condition',  categorical=True, random=True)
    mod5.add_term('subject', label='nipymc_randstim_subject', split_by='condition', categorical=True, random=True)
    groupA = str(mod5.level_map['nipymc_randstim_condition'][0])
    groupB = str(mod5.level_map['nipymc_randstim_condition'][1])
    mod5.add_deterministic('nipymc_randstim_cond_diff',
        "self.dists['b_nipymc_randstim_condition']["+groupB+"] - self.dists['b_nipymc_randstim_condition']["+groupA+"]")
    mod5.set_y('y', scale=None, detrend=False, ar=0 if ar is None else 2)
    print('fitting nipymc_randstim model...')
    mod5_fitted = mod5.run(samples=SAMPLES, verbose=False, find_map=False)

    # # save PNG of traceplot
    # plt.figure()
    # pm.traceplot(trace2[BURN:])
    # plt.savefig('pymc3_randstim.png')
    # plt.close()

    # plt.figure()
    # pm.traceplot(mod5_fitted.trace[BURN:])
    # plt.savefig('nipymc_randstim.png')
    # plt.close()

    ######################################
    ########## SAVE RESULTS ##############
    ######################################

    # return parameter estimates
    print('computing and returning parameter estimates...')

    # lists of traces and names of their model parameters
    traces = [trace0,            # fixstim
              trace1,            # nostim
              trace2,            # randstim
              mod3_fitted.trace, # nipymc_fixstim
              mod4_fitted.trace, # nipymc_nostim
              mod5_fitted.trace] # nipymc_randstim
    parlists = [[x for x in trace.varnames if 'log' not in x and 'u_' not in x] for trace in traces]

    # get posterior mean and SDs as lists of lists
    means = [[trace[param][BURN:].mean() for param in parlist] for trace, parlist in zip(traces, parlists)]
    SDs = [[trace[param][BURN:].std() for param in parlist] for trace, parlist in zip(traces, parlists)]

    # print list of summary statistics
    stats = sum([['posterior_mean']*len(x) + ['posterior_SD']*len(x) for x in parlists], [])
    print(stats)
    print(len(stats))

    # print parameter names in the order in which they are saved
    parlists = [2*parlist for parlist in parlists]
    extra_params = []
    params = [param for parlist in parlists for param in parlist] + extra_params
    print(params)

    # add SPM model results
    ans = [summary for model in zip(means, SDs) for summary in model]
    ans = [sub_diffs.mean(0).tolist(), (sub_diffs.std(0)/(len(sub_diffs.index)**.5)).tolist()] + ans
    params = ['SPM_cond_diff','SPM_A_mean','SPM_B_mean','SPM_AR1','SPM_AR2']*2 + params
    stats = ['posterior_mean']*5 + ['posterior_SD']*5 + stats

    # add test statistics for all models
    # grab all posterior means
    nums = [np.array(x) for x in ans][::2]
    # grab all posterior SDs
    denoms = [np.array(x) for x in ans][1::2]
    # divide them
    zs = [n/d for n,d in zip(nums,denoms)]
    zs = sum([x.tolist() for x in zs], [])
    # keep only the test statistics related to cond_diff
    labels = [params[i] for i in [j for j,x in enumerate(stats) if x=='posterior_mean']]
    zs = [(z,l) for z,l in zip(zs,labels) if 'cond_diff' in l]
    # add them to the results
    ans = [[x[0] for x in zs]] + ans
    params = [x[1] for x in zs] + params
    stats = ['test_statistic']*7 + stats

    # return the parameter values
    # for first instance only, also return param names and etc.
    if int(instance)==0: ans = [ans, params, stats]
    return ans

# run simulation
print('beginning simulation...')
dat = simulate(num_subs=int(p), num_stims=int(q), A_mean=1, B_mean=2, sub_A_sd=1,
    sub_B_sd=1, stim_A_sd=float(s), stim_B_sd=float(s), resid_sd=1,
    ar=[.45,.15], block_size=8)
print('sim complete. saving results...')

# write results to disk as pickle
# w = write, r = read, a = append
# b = binary
output = open('/scratch/03754/jaw5629/xsim_appendix/xsim_p'+str(p)+'_q'+str(q)+'_s'+str(s)+'_dat'+str(instance)+'.pkl', 'wb')
pickle.dump(dat, output)
output.close()

print('finished nicely!')
