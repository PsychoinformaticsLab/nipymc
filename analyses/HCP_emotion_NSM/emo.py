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
factory = nipymc.data.EmotionTaskFactory('../data')
dataset = factory.get_dataset(discard_vols=5)

# random stimulus model
emo1 = nipymc.model.BayesianModel(dataset)

# fixed effects
emo1.add_term('intercept')
emo1.add_term('emotion', categorical=True, scale=False)
# contrasts
emo1.add_deterministic('faceVsShape',
    "self.dists['b_emotion'][0] - T.mean(self.dists['b_emotion'][1:3])")
emo1.add_deterministic('angerVsFear',
    "self.dists['b_emotion'][1] - self.dists['b_emotion'][2]")

# random effects
# emo1.add_term(['left_stim','right_stim'], label='stimulus',
#               split_by='condition',random=True, categorical=True)
emo1.add_term('subject', split_by='emotion', random=True, categorical=True)

# DV
emo1.set_y('r'+region, scale='run', detrend=True, ar=2)

# Sample. verbose will spit out progress and find_MAP info. find_map can be
# turned on or off. Can also pass extra arguments onto the sampler,
# e.g., here I set the target_accept to something different.
fitted = emo1.run(samples=SAMPLES, verbose=True, find_map=False)

# save summary as pickle
summary = fitted.summarize(burn=BURN)
pickle.dump(summary, open('/corral-repl/utexas/neurosynth/results/emo_results/emo_fix_r'+region+'_summary.pkl', 'wb'))
pickle.dump(fitted.trace, open('/corral-repl/utexas/neurosynth/results/emo_results/emo_fix_r'+region+'_trace.pkl', 'wb'))

# save PNG of traceplot
plt.figure()
pm.traceplot(fitted.trace[BURN:])
plt.savefig('/corral-repl/utexas/neurosynth/results/emo_results/emo_fix_r'+region+'.png')

# print list of parameters to signify end of script
print(fitted.trace.varnames)
