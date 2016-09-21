import numpy as np
import pandas as pd
import sys, pickle, os

# first arg = number of parameters (p)
# args 1:p = names of parameters
# args p+1:2p = statistic type
# p = int(sys.argv[1])
# names = list(sys.argv[2:(p+2)])
# stats = list(sys.argv[(p+2):(2*(p+1)+1)])

# get list of all pickle files in current directory
filenames = [f for f in os.listdir('.') if os.path.isfile(f)]
filenames = [x for x in filenames if 'pkl' in x]

# get lists of param names and stat types from any dat0 file
dat0 = pickle.load(open(filenames[['dat0' in x for x in filenames].index(True)], 'rb'))
names = dat0[1]
stats = dat0[2]

for p in ['16', '32', '64']:
	for q in ['16', '32', '64']:
		for s in ['0.0001', '1', '2']:
			# unpickle results for this parameter combination
			results = [pickle.load(open(x, 'rb'))[0] if x == 'xsim_p'+p+'_q'+q+'_s'+s+'_dat0.pkl'
				else pickle.load(open(x, 'rb'))
				for x in filenames
				if x[:x.index('_dat')] == 'xsim_p'+p+'_q'+q+'_s'+s]

			# break out of inner loop if no results found
			if len(results) == 0:
				print('\nxsim_p'+p+'_q'+q+'_s'+s[0]+': '+str(len(results))+' iterations')
				break

			# convert to numpy array
			results = np.array([sum(x, []) for x in results])

			# print results header and filter extreme values
			print('\nxsim_p'+p+'_q'+q+'_s'+s[0]+': '+str(len(results))+' iterations, '+
				str((abs(results) > 100).sum())+' extreme values filtered')
			results[abs(results) > 100] = np.nan

			# print results
			pd.set_option('display.max_rows', 2*len(names))
			pd.set_option('display.float_format', lambda x: '%.2f' % x)
			print(pd.DataFrame({'param_name':names,
			                    'statistic':stats,
			                    'sim_mean':np.round(np.nanmean(results, axis=0), 2),
			                    'sim_SD':np.round(np.nanstd(results, axis=0), 2)
			                    }))
