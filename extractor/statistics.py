
from scipy.stats import skew, kurtosis
import numpy as np


def IQR(data):
    q3, q1 = np.percentile(data, [75, 25])
    return q3 - q1


def describe(data):
    # Describe data with statistics
    return {
            'mean': np.mean(data),
            'min': min(data),
            'max': max(data),
            'med': np.median(data),
            'IQR': IQR(data),
            'stdev': np.std(data),
            'skew': skew(data),
            'kurtosis': kurtosis(data),
    }
