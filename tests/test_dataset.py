from nipymc.data import Dataset
from nose.tools import (assert_equal, assert_is_instance)
from os.path import dirname, join
import pandas as pd


def test_dataset_initializes():
    data_dir = join(dirname(__file__), 'data')
    design = pd.read_csv(join(data_dir, 'test_design.txt'), sep=None)
    activation = pd.read_csv(join(data_dir, 'test_activation.txt'), sep=None)
    ds = Dataset(design, activation, 2.)
    assert_is_instance(ds, Dataset)
    assert_equal(ds.TR, 2)
    assert_equal(ds.n_vols, 4)
    assert_equal(ds.n_runs, 2)