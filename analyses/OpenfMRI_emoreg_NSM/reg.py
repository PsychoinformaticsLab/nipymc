import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, pickle, os
import pymc3 as pm
import nipymc
from nipymc import *
import pandas as pd

# 1st argument = which region to analyze
region = str(sys.argv[1])

# global variables...
SAMPLES = 3000
BURN = 1000

# get data
design = pd.read_csv('../data/REGULATION_behav/regulation_behav.txt', sep='\t')
activation = pd.read_csv('../data/REGULATION_behav/regulation_ts.txt', sep='\t')
dataset = nipymc.data.Dataset(design, activation, 2.0)

# random stimulus model
mod = nipymc.model.BayesianModel(dataset)

# fixed effects
mod.add_term('intercept')
mod.add_term('trial_type_orig', categorical=True)
# contrasts
expr = "self.dists['b_trial_type_orig'][1] - self.dists['b_trial_type_orig'][0]"
mod.add_deterministic('neg_vs_neut', expr=expr)
expr = "self.dists['b_trial_type_orig'][2] - self.dists['b_trial_type_orig'][1]"
mod.add_deterministic('suppr_vs_att', expr=expr)

# random effects
# mod.add_term('image_num', split_by='trial_type_orig', random=True, categorical=True)
mod.add_term('subject', split_by='trial_type_orig', random=True, categorical=True)

# DV
mod.set_y('r'+region, scale='run', detrend=True, ar=2)

# Sample. verbose will spit out progress and find_MAP info. find_map can be
# turned on or off. Can also pass extra arguments onto the sampler,
# e.g., here I set the target_accept to something different.
fitted = mod.run(samples=SAMPLES, verbose=True, find_map=False)

# save summary as pickle
summary = fitted.summarize(burn=BURN)
pickle.dump(summary, open('/corral-repl/utexas/neurosynth/results/reg_results/reg_fix_r'+region+'_summary.pkl', 'wb'))
pickle.dump(fitted.trace, open('/corral-repl/utexas/neurosynth/results/reg_results/reg_fix_r'+region+'_trace.pkl', 'wb'))

# save PNG of traceplot
plt.figure()
pm.traceplot(fitted.trace[BURN:])
plt.savefig('/corral-repl/utexas/neurosynth/results/reg_results/reg_fix_r'+region+'.png')

# print list of parameters to signify end of script
print(fitted.trace.varnames)
