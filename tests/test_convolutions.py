from nipymc.convolutions import get_convolution, gamma, spm_hrf
from nose.tools import (assert_equal, assert_true, assert_is_instance,
                        assert_raises, assert_is_not_none)
from numpy.testing import assert_array_equal


def test_get_convolution():
    ''' Check whether convolutions are retrieved appropriately. '''
    assert_raises(KeyError, get_convolution, 'fictional_hrf')
    conv1 = get_convolution('gamma', n=5)
    conv2 = gamma(n=5)
    assert_array_equal(conv1, conv2)

def test_gamma():
    duration = 10
    hz = 10
    conv = gamma(duration=duration, hz=hz)
    assert_equal(len(conv), duration*hz)

def test_spm():
    tr = 0.5
    p = [6, 16, 1, 1, 6, 0, 20]
    conv = spm_hrf(tr=0.5, p=p)
    assert_equal(len(conv), p[-1]/tr)