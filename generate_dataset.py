# -*- coding: utf-8 -*-
"""
Created on Tue Oct 02 10:23:49 2018

@author: Chris
"""

import numpy as np
import time
import sys

# (Vector count, vector length, filename)
datasets = [
            #( int(130e6), 300) # 16x GK210 max is ~128M
            ( int(128e6), 32) # Predicted max for 1x V100
            #( int(10e6), 4096) # 16x GK210 max should be ~9M                 
            #( int(210e6), 96) # 8x K80 max is ~200M
            #( int(100e6), 300) # 8x K80 max is ~64M     
            #( int(5e6), 4096) # 8x K80 max is ~4.7M
        ]

for ds in datasets:
    vec_count = ds[0]
    vec_len = ds[1]

    # =====================================
    #          Generate Dataset
    # =====================================    
    
    print('===================================================')
    print('Generating dataset %s x %d' % ("{:,}".format(vec_count), vec_len))
    print('===================================================')

    sys.stdout.flush()
    
    t0 = time.time()
                
    vecs = np.random.rand(vec_count, vec_len).astype('float32')
    
    print('   Done. Took %.2f seconds.' % (time.time() - t0))
        
    print('\nDataset is [%d x %d]\n' % vecs.shape)

    # =====================================
    #          Save Dataset
    # =====================================
    
    print('\nSaving dataset...')
    sys.stdout.flush()
    
    t0 = time.time()
    
    np.save('./%d_x_%d.npy' % (vec_count, vec_len), vecs)
    
    print('   Done. Took %.2f seconds.' % (time.time() - t0))

