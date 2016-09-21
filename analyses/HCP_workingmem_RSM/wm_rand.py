import matplotlib
matplotlib.use('Agg')
import time
import pymc3 as pm
import nipymc
from nipymc import *
import sys, pickle, gc
import matplotlib.pyplot as plt
sys.setrecursionlimit(10000)

# 1st argument = which region to analyze
region = str(sys.argv[1])

# global variables...
SAMPLES = 3000
BURN = 1000

# get data
factory = nipymc.data.WMTaskFactory('../data')
dataset = factory.get_dataset(top_stimuli=96, discard_vols=5)

# initialize the model
model1 = nipymc.model.BayesianModel(dataset)

# clean up
del factory
del dataset
gc.collect()

# add fixed effects
model1.add_term('intercept')
model1.add_term('nback_condition', categorical=True, scale=None)
#model1.add_term('stimulus_category', categorical=True, scale=None, orthogonalize=
#['nback_condition'])
model1.add_deterministic('0back_vs_2back',
    "self.dists['b_nback_condition'][0] - self.dists['b_nback_condition'][1]")

# add random effects
model1.add_term('subject', split_by='nback_condition', categorical=True, random=True)
#model1.add_term('subject', split_by='stimulus_category', categorical=True, random=True)
model1.add_term('stimulus', split_by='stimulus_category', categorical=True, random=True)

# Set the y variable and scale/detrend it
model1.set_y('r'+region, scale='run', detrend=True, ar=2)

# Sample
fitted = model1.run(samples=SAMPLES, verbose=True, find_map=False)

# save summary as pickle
summary = fitted.summarize(burn=BURN)
pickle.dump(summary, open('/corral-repl/utexas/neurosynth/results/wm_results/wm_rand_r'+region+'_summary.pkl', 'wb'))
pickle.dump(fitted.trace, open('/corral-repl/utexas/neurosynth/results/wm_results/wm_rand_r'+region+'_trace.pkl', 'wb'))

# save PNG of traceplot
plt.figure()
pm.traceplot(fitted.trace[BURN:])
plt.savefig('/corral-repl/utexas/neurosynth/results/wm_results/wm_rand_r'+region+'.png')

# print list of parameters to signify end of script
print(fitted.trace.varnames)
