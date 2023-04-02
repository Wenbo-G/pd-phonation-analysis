"""The need for this comes from the initially wrong calculations surrounding several features (spread1 and spread2). The development of the methods below were made in consultation with the original author of the data set. That is to say, these are now correct."""

from collections import Counter
import numpy as np
from scipy.stats import entropy
from statsmodels.tsa.ar_model import AutoReg

def remove_bad_samples(mPower_samples):
    """After inspecting the 20 most extreme samples for each feature (23 features, 20 samples, an upper bound of 4300 samples inspected), these were the features that we deemed low quality and should be removed from analysis"""

    #A list of the indices of bad samples
    bad_list = [30672,
                43792,
                48962,
                57255,
                5796,
                36194,
                20433,
                38254,
                53804,
                40194,
                2825,
                24893,
                35322,
                20906,
                39051,
                27306,
                20907,
                19185,
                25963,
                37076,
                15787,
                7357,
                31004,
                18473,
                17859,
                46000,
                55840,
                9699,
                14013,
                11524,
                5831,
                19690]

    for bad_ind in bad_list: mPower_samples = mPower_samples.drop(bad_ind)

    return mPower_samples
    
    
def adjust_feature_scale(mPower_samples):
    """The scale of several features between the mPower and Parkinsons data set were different (percentages were sometimes as XX.xx, instead of 0.XXxx). This corrects the scale"""

    features_to_adjust = ['Jitter (local) (%)','Jitter (rap) (%)','Jitter (ppq5) (%)','Jitter (ddp) (%)','Shimmer (local) (%)','Shimmer (apq3) (%)','Shimmer (apq5) (%)','Shimmer (apq11) (%)','Shimmer (dda) (%)']
    
    for feature in features_to_adjust: mPower_samples[feature] = mPower_samples[feature]/100
    
    return mPower_samples
    
    

def recalculate_spreads(mPower_samples,base=0.1,log_base=2):
    """As mentioned above, spread1 and spread2 were initially calculated wrong. This recalculates them correctly"""
    for ind in mPower_samples.index:
        f0list = mPower_samples.loc[ind,'f0 List']
        s1 = -11 + 12 * spread1(f0list,number_of_bins=401,normalised=True) #401 = 8*50+1 = num_of_halftones*50+1. Ensures 0 to be an interval. The "-11+12*" linearly scales similarly to OPD data set 
        s2 = spread1(f0list,number_of_bins=337,normalised=True) #337 = 28*12+1 = num_of_halftones * 12 + 1. This ensure 0 to be an interval
        
        mPower_samples.loc[ind,'spread1 (negative entropy of F0)'] = s1
        mPower_samples.loc[ind,'spread2 (standard error of F0)'] = s2
        
    return mPower_samples    
    
def quantise(array,bins,number_of_bins,return_idx=False):
    """Quantise array into bins by rounding to nearest bin
    
    Parameters:
        array (list-like): the array/list to quantise. Length n
        bin (list-like): the array/list to be quantised to. Allows nonlinear quantising. Length m
    """
    def scale(array_to_scale,bins,number_of_bins):
        scaled_array = (array_to_scale - min(bins)) / (max(bins) - min(bins)) * (number_of_bins - 1)
        return scaled_array
    
    
    array = np.array(array)
    array = scale(array,bins,number_of_bins)
    
    idx = array.round().astype(int)
    if return_idx: return idx
    else: return bins[idx]

def spread1(f0list,number_of_bins=60,normalised=False,log_base=None):
    """Correctly calculate spread1"""
    halftone_range = np.array(range(-5,3+1))
    return calc_entropy(f0list,halftone_range,number_of_bins,normalised,log_base)

def spread2(f0list,number_of_bins=60,normalised=False,log_base=None):
    """Correctly calculate spread2"""
    halftone_range = np.array(range(-14,14+1))
    return calc_entropy(f0list,halftone_range,number_of_bins,normalised,log_base)

def calc_entropy(f0list,halftone_range,number_of_bins,normalised,log_base):
    """Compute the entropy of quantised sustained phonation. Used for spread1 and spread2"""
    f0list = np.array(f0list)
    f0 = np.mean(f0list)
    halftone_series = 12*np.log2(f0list/f0)
    model = AutoReg(halftone_series,lags=1)
    trained_model = model.fit()
    quantised_bins = np.linspace(min(halftone_range),max(halftone_range),num=number_of_bins)
    residuals = np.clip(trained_model.resid,a_min=min(halftone_range),a_max=max(halftone_range))    
    quantised_residuals = quantise(residuals,quantised_bins,number_of_bins)
    c = Counter(quantised_residuals)
    if normalised:
        return entropy(list(c.values()),base=log_base)/entropy([1]*number_of_bins,base=log_base)
    else:
        return entropy(list(c.values()),base=log_base)



