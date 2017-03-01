import numpy as np
from scipy.misc import factorial
from scipy import stats


def get_convolution(name, **kwargs):
    return globals()[name](**kwargs)


def gamma(duration=20, hz=10, tau=1.25, n=3):
    t = np.linspace(0, duration, duration*hz)
    return (((t/tau)**(n-1))*np.exp(-(t/tau))) / factorial(tau*(n-1))


def spm_hrf(tr, p=[6, 16, 1, 1, 6, 0, 32]):
    """ An implementation of spm_hrf.m from the SPM distribution

    Arguments:

    Required:
    tr: repetition time at which to generate the HRF (in seconds)

    Optional:
    p: list with parameters of the two gamma functions:
                                                         defaults
                                                        (seconds)
       p[0] - delay of response (relative to onset)         6
       p[1] - delay of undershoot (relative to onset)      16
       p[2] - dispersion of response                        1
       p[3] - dispersion of undershoot                      1
       p[4] - ratio of response to undershoot               6
       p[5] - onset (seconds)                               0
       p[6] - length of kernel (seconds)                   32
    """

    p = [float(x) for x in p]

    fMRI_T = 16.0

    tr = float(tr)
    dt = tr/fMRI_T
    u = np.arange(p[6]/dt + 1) - p[5]/dt
    hrf = stats.gamma.pdf(u, p[0]/p[2], scale=1.0/(dt/p[2])) - \
        stats.gamma.pdf(u, p[1]/p[3], scale=1.0/(dt/p[3]))/p[4]
    good_pts = np.array(range(np.int(p[6]/tr)))*fMRI_T
    hrf = hrf[good_pts.astype(int)]
    hrf = hrf/np.sum(hrf)
    return hrf
