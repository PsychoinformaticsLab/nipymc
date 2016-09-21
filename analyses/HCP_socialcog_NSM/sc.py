import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, pickle, os
import pymc3 as pm
import nipymc
from nipymc import *

# 1st argument = which region to analyze
region = str(sys.argv[1])

# global variables...
SAMPLES = 3000
BURN = 1000

# get data
factory = nipymc.data.SocialTaskFactory('../data')
dataset = factory.get_dataset(top_stimuli=10, discard_vols=5)

# random stimulus model
mod = nipymc.model.BayesianModel(dataset)

# fixed effects
mod.add_term('intercept')
mod.add_term('condition', categorical=True, scale=False)
# contrasts
mod.add_deterministic('randomVsMental',
    "self.dists['b_condition'][0] - self.dists['b_condition'][1]")

# random effects
# mod.add_term('stimulus', split_by='condition', random=True, categorical=True)
mod.add_term('subject', split_by='condition', random=True, categorical=True)

# DV
mod.set_y('r'+region, scale='run', detrend=True, ar=2)

# Sample. verbose will spit out progress and find_MAP info. find_map can be
# turned on or off. Can also pass extra arguments onto the sampler,
# e.g., here I set the target_accept to something different.
fitted = mod.run(samples=SAMPLES, verbose=True, find_map=False)

# save summary as pickle
summary = fitted.summarize(burn=BURN)
pickle.dump(summary, open('/corral-repl/utexas/neurosynth/results/sc_results/sc_fix_r'+region+'_summary.pkl', 'wb'))
pickle.dump(fitted.trace, open('/corral-repl/utexas/neurosynth/results/sc_results/sc_fix_r'+region+'_trace.pkl', 'wb'))

# save PNG of traceplot
plt.figure()
pm.traceplot(fitted.trace[BURN:])
plt.savefig('/corral-repl/utexas/neurosynth/results/sc_results/sc_fix_r'+region+'.png')

# print list of parameters to signify end of script
print(fitted.trace.varnames)
