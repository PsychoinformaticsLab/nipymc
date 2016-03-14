from abc import abstractmethod, ABCMeta
import json
from six import string_types
from glob import glob
from collections import defaultdict
from os.path import join, basename, exists
import numpy as np
import pandas as pd
import re
from .model import BayesianModel


class Dataset(object):

    def __init__(self, design, activation, TR):
        self.design = design
        self.activation = activation
        self.TR = TR
        self.n_vols = self.activation['vol'].nunique()
        self.n_runs = self.activation['run'].nunique()


class HCPDatasetFactory(object):

    __metaclass__ = ABCMeta

    def __init__(self, data_path='../data'):

        self.data_path = data_path
        config_file = join(data_path, 'configs', self.task + '.json')
        self.ts_path = join(data_path, self.task + '_timeseries')
        self.config = json.load(open(config_file))
        self.n_vols = self.config['number_of_volumes_per_run']
        self.n_runs = self.config['number_of_runs']
        self._load_timeseries()
        self.eprime_data, self.ev_data = self._get_behav_data()
        self._load_design()

    def get_dataset(self, runs=None, subjects=None, top_stimuli=None,
                    discard_vols=5):
        '''
        Args:
            runs (list): list of runs to include in the Dataset, indexed from
                0. If None (default), all available runs are included.
            subjects (int, list, None): subjects to include in the Dataset.
                If an int is passed, the first n subjects will be used. If a
                list, must be the IDs of the subjects to keep. If None, all
                subjects in the search path will be included.
            top_stimuli( int): optionally, the number of stimuli (sorted by
                frequency of presentation) to consider valid. All runs or
                subjects who were presented with at least one stimulus not in
                this set will be removed from the dataset to ensure reliable
                estimation of individual stimulus effects. 
            discard_vols (int): Number of volumes at the beginning of each
                run to discard in order to allow for signal stabilization.
        '''

        ts = self.timeseries_data.copy()
        design = self.design_data.copy()
        tr = self.config['TR']

        # Select runs
        if runs is not None:
            design = design.query('run in @runs')
        else:
            runs = design['run'].unique()

        # Select subjects
        if subjects is not None:
            if isinstance(subjects, int):
                subjects = design['subject'].unique()[:subjects]
            design = design.query('subject in @subjects')

        # Drop TRs at beginning of run
        if discard_vols:
            ts = ts.query('vol > @discard_vols')
            ts['vol'] -= discard_vols
            window = self.config['TR'] * discard_vols
            design = design.query('run_onset >= @window')
            design['onset'] -= window * (design['run']+1)
            design['run_onset'] -= window

        # Drop all runs/subjects who have scanning runs with a non-standard
        # design--i.e., where some stimuli other than the 96 normal ones were
        # presented. This is necessary in order to obtain reliable stimulus-
        # level estimates, otherwise the model goes to shit when adding
        # stimulus as a random effect.
        if top_stimuli is not None:
            stim_counts = design['stimulus'].value_counts()
            valid_stims = set(list(stim_counts[:top_stimuli].index))
            # Identify subjects with at least one stim not in the valid list
            bad_subjects = design.groupby('subject', as_index=False).filter(
                lambda x: bool(set(list(x['stimulus'].unique())) - valid_stims)
            )['subject'].unique()
            design = design.query('subject not in @bad_subjects')
            ts = ts.query('subject not in @bad_subjects')

        ### Validation ###

        class ValidationError(Exception):
            pass

        # Drop subjects with missing runs
        design = design.groupby('subject').filter(
            lambda x: x['run'].nunique() == len(runs))

        # Drop subjects with unequal number of TRs in any run
        sub_vols = design.groupby('subject')['run'].count()
        # Assume modal value is correct; drop the rest. TODO: do some extra
        # validation and issue warning or exception if there are multiple
        # frequent values.
        if sub_vols.nunique() > 1:
            mode = sub_vols.value_counts()[0]
            bad_subs = sub_vols[sub_vols != mode].index.tolist()
            design = design.query('subject not in @bad_subs')

        # Fail if any onsets are outside the bounds of activation data
        if (design['onset']/tr > len(ts)).any():
            raise ValidationError("At least one event onset in the dataset "
                                  "occurs outside the bounds of the fMRI data.")

        # Only keep subjects with both behav and activation data
        des_subs, ts_subs = design['subject'].unique(), ts['subject'].unique()
        common = list(set(des_subs) & set(ts_subs))
        if len(des_subs) != len(common):
            design = design.query('subject in @common')
        if len(ts_subs) != len(common):
            ts = ts.query('subject in @common')

        # Make sure both DFs are properly sorted in same order
        ts = ts.sort_values(['subject', 'run', 'vol'])
        design = design.sort_values(['subject', 'run', 'onset'])

        return Dataset(design, ts, TR=tr)

    @abstractmethod
    def _preprocess(self):
        ''' Must be implemented in every subclass. '''
        pass

    def _load_timeseries(self):
        task_path = join(self.data_path, '%s_timeseries' % self.task, '*.txt')
        run_files = glob(task_path)

        # Read in all timeseries data and store in pandas DF
        data = []
        for f in run_files:
            sub, run = basename(f).replace('.txt', '').split('_')
            _run = pd.read_csv(f, sep='\s+', header=None)
            _run.columns = ['r%d' % (i+1) for i in _run.columns]
            _run['vol'] = np.arange(len(_run)) + 1
            _run['subject'] = int(sub)
            _run['run'] = int(run)
            data.append(_run)
        self.timeseries_data = pd.concat(data, axis=0)

    def _read_text_files(self, path, header=0):
        files = glob(path)
        data = []
        for f in files:
            try:
                _run = pd.read_csv(f, sep='\t', header=header)
            except:
                continue
            subject, run_num = basename(f).split('_')[1:3]
            subject = int(subject)
            run_num = 1 if run_num == 'RL' else 2
            filename = basename(f).replace('.txt', '')
            _run['subject'] = subject
            _run['run'] = run_num - 1
            _run['filename'] = filename
            data.append(_run)
        return pd.concat(data, axis=0)

    def _get_behav_data(self):
        eprime_path = join(
            self.data_path, 'eprime', '%s_*_??_TAB.txt') % self.task
        ep = self._read_text_files(eprime_path)
        ev_path = join(self.data_path, 'onsets', '%s_*.txt') % self.task
        ev = self._read_text_files(ev_path, header=None)
        ev = ev.rename(columns={0: 'onset', 1: 'duration', 2: 'amplitude'})
        ev['filename'] = ev['filename'].str.replace('^.*[RL]{2}_', '')
        return (ep, ev)

    def _load_design(self):
        ''' Generic preprocessing of behavioral data that can be applied to
        all HCP tasks (provided JSON config vars are appropriately set.)
        '''

        ep = self.eprime_data

        # Task-specific filtering/preprocessing
        if hasattr(self, '_preprocess'):
            ep = self._preprocess(ep)

        # Trim E-Prime data to just the columns we need and rename for brevity
        keep_cols = self.config['eprime_columns']
        ep = ep[list(keep_cols.keys()) + ['subject', 'run', 'filename']]
        ep = ep.rename(columns=keep_cols)

        # Onsets can be extracted either from behav files, or from event files
        if 'scanner_sync' in self.config and 'onset' in ep.columns:
            ep['onset'] = (ep['onset'] - ep['sync'])/1000.
            ep['run_onset'] = ep['onset']
            ep['onset'] += ep['run'] * self.n_vols * self.config['TR']
            trial_data = ep
        else:
            # Keep onsets for only specific event types
            event_types = self.config['filter_by_events']
            onsets = self.ev_data.query('filename in @event_types')

            # Make onset times relative to start of first run, not each run
            onsets['run_onset'] = onsets['onset']
            onsets['onset'] += onsets['run'] * self.n_vols * self.config['TR']
            onsets = onsets.sort_values('onset')

            # Add run-level numerical index to both DFs and merge
            onsets['index'] = onsets.groupby(['subject', 'run']).cumcount()
            ep['index'] = ep.groupby(['subject', 'run']).cumcount()
            trial_data = ep.merge(onsets, on=['subject', 'run', 'index'],
                                  how='left')

        # Filter out subjects with events that don't line up perfectly
        run_grps = trial_data.groupby(['subject', 'run'])
        self.design_data = run_grps.filter(
            lambda x: x['onset'].notnull().all())

        # Do any task-specific post-processing
        self._postprocess()


class WMTaskFactory(HCPDatasetFactory):

    task = 'WM'

    def _preprocess(self, data):
        return data[data['Stimulus[Block]'].notnull() &
                    ~data['Procedure[Block]'].str.contains('Cue0Back')]

    def _postprocess(self):
        # Track number of times each stimulus has been previously presented
        reps = self.design_data.groupby(['subject', 'stimulus']).cumcount()
        self.design_data['repetition'] = reps


class SocialTaskFactory(HCPDatasetFactory):

    task = 'SOCIAL'

    def _preprocess(self, data):
        return data.query('Procedure == "SOCIALrunPROC"')

    def _postprocess(self):
        pass


class EmotionTaskFactory(HCPDatasetFactory):

    task = 'EMOTION'

    def _preprocess(self, data):
        def sync_onsets(grp):
            grp['SyncSlide.OnsetTime'] = grp['SyncSlide.OnsetTime'].min()
            return grp
        data = data.groupby(['subject', 'run']).apply(sync_onsets)
        return data.query('Procedure == "TrialsPROC"')

    def _postprocess(self):
        stim = self.design_data['top_stim'].str.split('.').str.get(0)
        cols = self.design_data[['top_stim', 'left_stim', 'right_stim']]
        self.design_data['stimulus'] = cols.apply(lambda x: '_'.join(x), axis=1)
        self.design_data['condition'] = 'shape'
        face = stim.apply(lambda x: '_' in x)
        self.design_data.ix[face, 'condition'] = 'face'
        self.design_data['duration'] = 3
        reps = self.design_data.groupby(['subject', 'stimulus']).cumcount()
        self.design_data['repetition'] = reps
        self.design_data['emotion'] = 0
        self.design_data['emotion'][stim.str.get(1) == 'A'] = 1
        self.design_data['emotion'][stim.str.get(1) == 'F'] = -1
