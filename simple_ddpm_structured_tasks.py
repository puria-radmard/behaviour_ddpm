"""
4.12.24 - These are tasks which have multiple stages, and require inputs to change over time

The most simple example is a case where two conjunctive items are fed in at the same time, then a cue is fed in
    The noising process can just be a a plateauing function, which stops increasing at the stimulus change time
    The denoising process (as described above) has one input (all stimuli, all features) when the noise magnitude is plateaued,
        and another input (one stimulus, probe feature) when the noise is quenched.
    
    The training loss remains the string of denoising KLs
    The "desired" p(y|Z) is for now always at the correct item, i.e. no swaps are taught

    The unique part of this network is that it is trained probabilistically
    
    XXX we can directly compare this to:
        a. networks trained on just the final MSE to the item
        b. networks trained with a swapping p(y|Z)
    
For now, we start with a task which is just two features, each taking 8 values on the unit circle
"""

import torch
import numpy as np

from purias_utils.multiitem_working_memory.util.circle_utils import polar2cart


NUM_POSSIBLE_FEATURE_VALUES = 8
# NUM_FEATURE_DIMENSIONS = 2        No option for no - just a cue and a probe dimension
NUM_ITEMS = 3


def generate_simple_conjunctive_binding_task_trial(n_samples):
    """
    Randomly generate NUM_ITEMS of 2 feature dimensions each, with feature values \in [0, NUM_POSSIBLE_FEATURE_VALUES)

    XXX No swaps at the moment!

    responses:
        Right now, just the corret report dimension response (see below)

    network_input_loading and network_input_recall:
        The input is D = (NUM_ITEMS * 2) + 1 dimensional
        
        During the "loading" phase (see documentation at the top of the file)
            The first NUM_ITEMS dimensions is the angular coordinate of the probe dimension
            The next NUM_ITEMS dimensions is the angular coordinate of the first item
            The final dimension is zero
        
        During the "recall" phase
            The final dimension is the cued probe item
            All other dimensions are zero
    
    network_input_idxs:
        This is only for visualisation
        This is the report dimension feature, as the cue feature can be represented spatially (see responses)
    """
    probe_features = np.random.choice(NUM_POSSIBLE_FEATURE_VALUES, NUM_ITEMS, replace = False)
    probe_features_angular = -np.pi + 2 * np.pi * (probe_features / NUM_POSSIBLE_FEATURE_VALUES)            # [NUM_ITEMS]
    report_features = np.random.choice(NUM_POSSIBLE_FEATURE_VALUES, NUM_ITEMS, replace = False)
    report_features_angular = -np.pi + 2 * np.pi * (report_features / NUM_POSSIBLE_FEATURE_VALUES)          # [NUM_ITEMS]

    cued_item_idx = np.random.choice(NUM_POSSIBLE_FEATURE_VALUES, n_samples, replace = True)                    # [n_samples]
    import pdb; pdb.set_trace()
    cued_probe = probe_features_angular[cued_item_idx,None]     # [n_samples, 1]

    network_input_loading = np.concatenate([probe_features_angular, report_features_angular, np.zeros([1])])
    network_input_recall = np.concatenate([np.zeros(n_samples, 2 * NUM_ITEMS), cued_probe])

    network_input_idxs = report_features

    responses = polar2cart(report_features_angular)

    return responses, (network_input_loading, network_input_recall), network_input_idxs


# (function, num epochs, input size)
WM_TASK_INFORMATION_DICTIONARY = {
    'simple_conjunctive_binding_task': (generate_simple_conjunctive_binding_task_trial, 2, NUM_ITEMS * 2 + 1),
}

