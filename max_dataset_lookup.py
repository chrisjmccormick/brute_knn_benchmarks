# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 10:17:25 2018

@author: Chris
"""

import pandas as pd

test = {
        # Specify properties of the GPU
        'gpu_name': "Tesla K80",
        #'gpu_name': "Tesla V100",
        'n_gpu': 8,
        'gpu_RAM': 8 * 12,
        'faiss_v': '1.4.0', 
        'vec_len': 300
    }

# Select the trials matching this test's parameters.
df = pd.read_csv('max_dataset_size.csv')
df = df.loc[(df.gpu_name == test['gpu_name']) &
            (df.n_gpu == test['n_gpu']) &
            (df.gpu_RAM == test['gpu_RAM']) &
            (df.faiss_v == test['faiss_v']) &
            (df.vec_len == test['vec_len'])]

# Display the configuration properties.
print('For %dx %s with %d GB RAM:' % (test['n_gpu'], test['gpu_name'], int(test['gpu_RAM'])))

# Test for empty (no trials run)
if df.empty:
    print('ERROR - No capacity data found for this configuration')
# Test for no trials with success.
elif df.loc[df.success == True].empty:
    print('ERROR - No successful trials for this configuration')
else:
    # Find the largest success and smallest failure.
    max_success = int(df.loc[df.success == True].vec_count.max())
    min_fail = int(df.loc[df.success == False].vec_count.min())

    print('  Max Success: %s x %d' % ("{:,}".format(max_success), test['vec_len']))
    print('  Min Failure: %s x %d' % ("{:,}".format(min_fail), test['vec_len']))