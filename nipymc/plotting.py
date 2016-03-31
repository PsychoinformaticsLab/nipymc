from nilearn import plotting
from nilearn.image import new_img_like
import numpy as np
from glob import glob
import nibabel as nb
import re
from os.path import basename
import pickle
import pymc3 as pm


def plot_wholebrain_test_stat(path, roi_img, name, type='contrast', l1=0, l2=1,
                              plot=True, save=None, burn=0.5, verbose=False,
                              invert=False, **kwargs):
    ''' Plot cluster-by-cluster z-scores across the whole brain for a
    paired-samples contrast between the named effects n1 and n2.
    Args:
            path (str): path to all pickled PyMC3 trace results (one file per
                    region or cluster).
            roi_img (str): path to map containing cluster/region labels.
            name (str): name of factor whose levels we want to compare.
            type (str): whether to search for the named factor as a 'contrast'
                or as a 'term'.
            l1 (int): index of first level to compare
            l2 (int): index of second level to compare
            plot (bool): Whether or not to plot the generated image.
            save (str): Optional filename to write the resulting image to.
            burn (float or int): burn-in (i.e., number of samples to drop
                before comparison). If a float, interpreted as a proportion of
                all available samples. If an int, interpreted as the number of
                samples to drop.
            invert (bool): whether to invert the sign of the test statistic
            kwargs (dict): Any additional keywords to pass on to the nilearn
                    plot_stat_map function.
    '''

    roi_img = nb.load(roi_img)          # Load label image
    label_map = roi_img.get_data().round().astype(int)
    result = np.zeros(roi_img.shape)        # Empty volume for results

    # Loop over region trace files
    roi_traces = glob(path)

    for rt in roi_traces:

        label = int(re.search('(\d+)', basename(rt)).group(1))
        obj = pickle.load(open(rt, 'rb'))

        if hasattr(obj, 'trace'):

            trace = obj.trace

            if not hasattr(trace, name):
                raise AttributeError("The MultiTrace object does not contain the "
                                     "variable '%s'. Please make sure you have "
                                     "the right name." % name)

            if isinstance(burn, float):
                _burn = round(len(trace[name]) * burn)
            else:
                _burn = burn

            samp1 = trace[name][_burn:, l1]
            if len(samp1.shape) > 1:
                samp1 = samp1.mean(1)

            if l2 is None:
                mu = samp1
            else:
                samp2 = trace[name][_burn:, l2]
                if len(samp2.shape) > 1:
                    samp2 = samp2.mean(1)
                mu = samp1 - samp2

            z = mu.mean() / mu.std()

        else:
            # if name not in obj['contrasts']:
            if False:
                raise AttributeError("The trace summary object does not contain the "
                                     "variable '%s'. Please make sure you have "
                                     "the right name." % name)
            if type == 'contrast':
                z = obj['contrasts'][name][2]
            else:
                z = obj['terms'][name]['z']

        # Assign value to the correct voxels in the output image
        if invert:
            z *= -1
        result[label_map == label] = z

    # Write out image
    output_img = new_img_like(roi_img, result, roi_img.affine)
    if save is not None:
        output_img.to_filename(save)

    # Plot
    if plot:
        plotting.plot_stat_map(output_img, **kwargs)

    return output_img
