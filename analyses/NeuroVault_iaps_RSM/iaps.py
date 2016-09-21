import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, pickle, os
import pymc3 as pm
import nipymc
from nipymc import *
import pandas as pd
from theano import shared

# 1st argument = which region to analyze
region = str(sys.argv[1])

# global variables...
SAMPLES = 3000
BURN = 1000

# get data
activation = pd.read_csv('../data/extra/IAPS_activation.csv', index_col=0)
ratings = pd.read_csv('../data/extra/IAPS_ratings.csv')
data = ratings.merge(activation, left_on='Files', how='inner', right_index=True)

X = data.copy()
recode_vars = ['Valence', 'Subject', 'Picture', 'SEX', 'RACE']
for v in recode_vars:
    print("Processing %s... (%d values)" % (v, len(X[v].unique())))
    vals = X[v].unique()
    repl_dict = dict(zip(vals, list(range(len(vals)))))
    col = [repl_dict[k] for k in list(X[v].values)]
    X[v] = col

# Standardize ratings within-subject
X['rating'] = X.groupby('Subject')['Rating'].apply(lambda x: (x - x.mean())/x.std()).values

# random stimulus model
model = pm.Model()
with model:
    
    # Intercept
    mu = pm.Normal('intercept', 0, 10)
#     mu = 0
    
    # Categorical fixed effects
    betas = {}
#     cat_fe = ['Valence', 'RACE']
    cat_fe = ['Valence']
    for cfe in cat_fe:
        dummies = pd.get_dummies(X[cfe], drop_first=True).values
        _b = pm.Normal('b_%s' % cfe, 0, 10, shape=dummies.shape[1])
        mu += pm.dot(dummies, _b)
        betas[cfe] = _b

    # Continuous fixed effects
#     cont_fe = ['AGE', 'YRS_SCH', 'SEX']
    cont_fe = []
    for cfe in cont_fe:
        _b = pm.Normal('b_%s' % cfe, 0, 10)
        mu += _b * X[cfe].values
        betas[cfe] = _b
    
#     # Contrast between conditions
#     b_valence = betas['Valence']
#     c = pm.Deterministic('valence_contrast', b_valence[0] - b_valence[1])
    
    # Random effects
    for var in ['Subject', 'Picture']:
        n_levels = X[var].nunique()
        sigma = pm.HalfCauchy('sigma_%s' % var, 10)
        u = pm.Normal('u_%s' % var, mu=0, sd=sigma, shape=n_levels)
        mu += u[X[var].values]    
       
        # Random slopes
        if var=='Subject':
            sigma = pm.Uniform('sigma_%s_valence' % var, 0, 10)
            u = pm.Normal('u_%s_valence' % var, mu=0, sd=sigma, shape=n_levels)
            mu += u[X[var].values] * X['Valence'].values
 
    sigma = pm.HalfCauchy('sigma', 10)
    y = shared(X['r'+region].values)
    y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)
    
    nuts = pm.NUTS()
    trace = pm.sample(SAMPLES, step=nuts)

# save summary as pickle
fitted = nipymc.model.BayesianModelResults(trace)
summary = fitted.summarize(burn=BURN)
pickle.dump(summary, open('/scratch/03754/jaw5629/iaps_results/iaps_rand_r'+region+'_summary.pkl', 'wb'))
pickle.dump(trace, open('/scratch/03754/jaw5629/iaps_results/iaps_rand_r'+region+'_trace.pkl', 'wb'))

# save PNG of traceplot
plt.figure()
pm.traceplot(trace[BURN:])
plt.savefig('/scratch/03754/jaw5629/iaps_results/iaps_rand_r'+region+'.png')

# print list of parameters to signify end of script
print(trace.varnames)
