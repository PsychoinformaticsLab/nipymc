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
SAMPLES = 2000
BURN = 500

# get data
factory = nipymc.data.LanguageTaskFactory('../data')
keep_subs = [100307, 100408, 101006, 101107, 101309, 101915, 103111, 103414,
       103818, 105014, 105115, 106016, 108828, 110411, 111312, 111716,
       113619, 113922, 114419, 115320, 116524, 117122, 118528, 118730,
       118932, 120111, 122317, 122620, 123117, 123925, 124422, 125525,
       126325, 127630, 127933, 128127, 128632, 129028, 130013, 130316,
       131217, 131722, 133019, 133928, 135225, 135932, 136833, 138534,
       139637, 140925, 144832, 146432, 147737, 148335, 148840, 149337,
       149539, 149741, 151223, 151526, 151627, 153025, 154734, 156637,
       159340, 161731, 162733, 163129, 176542, 178950, 189450, 190031,
       192540, 196750, 198451, 199655, 201111, 208226, 211417, 211720,
       212318, 214423, 221319, 239944, 245333, 280739, 298051, 366446,
       397760, 414229, 499566, 654754, 672756, 756055, 792564, 856766,
       857263]
dataset = factory.get_dataset(discard_vols=5, subjects=keep_subs)

# group stimuli with <3 responses into a single 'dummy' stimulus
sc = dataset.design['stimulus'].value_counts()
inds = sc[sc < 3].index
dataset.design.ix[dataset.design['stimulus'].isin(inds), 'stimulus'] = 'dummy'

# # Add stimulus statistics
# ss = pd.read_csv('../data/extra/math_stim_stats.txt', sep='\t')
# merged = dataset.design.merge(ss, on='stimulus', how='left')

# # Add individual differences
# indiv = pd.read_csv('../data/extra/HCP_behav_data.csv')[['Subject', 'ReadEng_Unadj', 'PMAT24_A_CR']]
# indiv.columns = ['subject', 'reading', 'pmat']
# merged = merged.merge(indiv, on='subject', how='left')

# # Extract math level from question
# merged['level'] = merged['stimulus'].str.extract('math-level(\d+)')
# # merged['log_first'].hist(bins=40)

# for var in ['level', 'reading', 'log_first', 'pmat']:
#     v = merged[var].convert_objects(convert_numeric=True)
# #     print(v.value_counts(), len(v), np.sum(v.isnull()))
#     v = (v - v.mean())/ v.std()
#     v[v.isnull()] = 0
#     merged[var] = v
# #     print("HERE", var)

# dataset.design = merged

# # get starting values from summary of fixed stimulus model
# summary = pickle.load(open('/corral-repl/utexas/neurosynth/results/lang_results/lang_fix_r'+region+'_summary.pkl', 'rb'))
# start = {x:summary['terms'][x]['m'] for x in summary['terms'].keys()}

# random stimulus model
model = nipymc.model.BayesianModel(dataset)

# fixed effects
model.add_term('intercept')
model.add_term('condition', categorical=True, scale=False)
math = str(model.level_map['condition']['Math'])
story = str(model.level_map['condition']['Story'])
model.add_deterministic('storyVsMath',
    "self.dists['b_condition']["+math+"] - self.dists['b_condition']["+story+"]"
)
# model.add_term('reading', split_by='condition')
# model.add_term('pmat', split_by='condition')
# model.add_term('log_first')

# random effects
# model.add_term('stimulus', split_by='condition', random=True, categorical=True)
model.add_term('subject', split_by='condition', random=True, categorical=True)

# DV
model.set_y('r'+region, scale='run', detrend=True, ar=2)

# Sample. verbose will spit out progress and find_MAP info. find_map can be
# turned on or off. Can also pass extra arguments onto the sampler,
# e.g., here I set the target_accept to something different.
fitted = model.run(samples=SAMPLES, verbose=True, find_map=False)

# save summary as pickle
summary = fitted.summarize(burn=BURN)
pickle.dump(summary, open('/scratch/03754/jaw5629/lang_results_final/lang_fix_r'+region+'_summary.pkl', 'wb'))
pickle.dump(fitted.trace, open('/scratch/03754/jaw5629/lang_results_final/lang_fix_r'+region+'_trace.pkl', 'wb'))

# save PNG of traceplot
plt.figure()
pm.traceplot(fitted.trace[BURN:])
plt.savefig('/scratch/03754/jaw5629/lang_results_final/lang_fix_r'+region+'.png')

# print list of parameters to signify end of script
print(fitted.trace.varnames)
