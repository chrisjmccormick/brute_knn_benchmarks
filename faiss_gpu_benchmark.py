from __future__ import division

import time
from datetime import datetime
import sys
import numpy as np
import faiss
import pandas as pd
import os


'''
* Create a GitHub repo to house the code and results.
* Show results with different:
    X vector length - 96, 300, 4096
    * dataset vector count
    * batch size
    * k
    * Card type
    * Card count
    * Distance metric
    * Instance name

* Additionally, record in the .csv file:
   * Total latency
   * Total board memory  
   * NO - Hourly cost
   * Experiment repetitions
   * Dataset load time
   * Measurement date

* Create a gaussian kernel regression model to predict performance using feature scaling.
'''

ngpus = [1, 8, 16]
vec_lengths = [96, 300, 4096]


test = {
        # Specify properties of the GPU
        'gpu_name': "Tesla K80",
        'n_gpu': 2,
        'gpu_RAM': 2 * 12,
        'faiss_v': '1.4.0', 
        'instance': 'p2.8xlarge',
        
        # Specify some test properties
        'metric': 'L2'
    }

dataset_sizes = [(int(1e6), 96),
                 (int(10e6), 96),
                 (int(25.07e6), 96), # Max for 1x K80
                 
                 (int(50e6), 96),
                 (int(50.14e6), 96), # Max for 2x K80
                 
                 #(int(100e6), 96),
                 #(int(200e6), 96) # Max for 8x K80
                 
                 (int(1e6), 300),
                 (int(5e6), 300),
                 (int(8.08e6), 300), # Max for 1x K80
                 (int(10e6), 300),
                 
                 (int(100e5), 4096),
                 (int(590e5), 4096), # Max for 1x K80
                 (int(1e6), 4096)
                ]
           
batch_sizes = [1, 16, 64, 256, 1024]
ks = [1, 10, 100]
repetitions = 10

# For each dataset size...
for ds_shape in dataset_sizes:

    print('===================================================')
    print('Running benchmark with dataset %s x %d' % ("{:,}".format(ds_shape[0]), ds_shape[1]))
    print('===================================================')
    
    # Create a test object for storing all parameters and results of this experiment
    test['dataset_count'] = ds_shape[0]
    test['vec_len'] = ds_shape[1]
    
    # =====================================
    #          Generate Dataset
    # =====================================

    # Record the name of the dataset.
    test['dataset_name'] = 'random'
    
    print('Generating dataset...')
    sys.stdout.flush()
    
    t0 = time.time()
                
    vecs = np.random.rand(ds_shape[0], ds_shape[1]).astype('float32')
        
    print('   Done. Took %.2f seconds.' % (time.time() - t0))
        
    print('Dataset is [%d x %d]' % vecs.shape)
    
    # =====================================
    #            FAISS Setup
    # =====================================
    
    # Build a flat (CPU) index
    cpu_index = faiss.IndexFlatL2(vecs.shape[1])
    
    # Print the number of available GPUs. 
    print('Number of available GPUs: %d    Using: %d' % (faiss.get_num_gpus(), test['n_gpu']))
    
    # Enable sharding so that the dataset is divided across the GPUs rather than
    # replicated.
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    
    # Make it into a gpu index
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=test['n_gpu'])
    
    # Add vecs to our GPU index
    
    print('Adding dataset to index...')
    sys.stdout.flush()

    t0 = time.time()    
    
    gpu_index.add(vecs)
 
    elapsed = time.time() - t0
    print('Building index took %.2f seconds' % (elapsed))
    
    test['load_time'] = elapsed
            
    # =====================================
    #            Benchmark
    # =====================================

    # Warm up query.
    D, I = gpu_index.search(vecs[:16], k=100) 
           
    tests = []        
    
    # Run queries with different batch sizes and 'k' values. 
    for batch_size in batch_sizes:
        for k in ks:
            
            print('Running batch %d with k=%d...' % (batch_size, k))
            sys.stdout.flush()
            
            test["batch_size"] = batch_size
            test["k"] = k
            test["repetitions"] = repetitions
            test['timestamp'] = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            
            # Run the query multiple times. We'll record the min, max, and
            # average latency across the repetitions.           
            run_times = []
            
            for run in range(repetitions):
                t0 = time.time()
                
                D, I = gpu_index.search(vecs[:batch_size], k) 
                
                run_times.append(time.time() - t0)

            test['latency'] = np.mean(run_times)
            test['min_latency'] = min(run_times)
            test['max_latency'] = max(run_times)
            
            print('  Min: %.3f sec, Max: %.3f, Avg: %.3f' % (min(run_times), max(run_times), np.mean(run_times)))
            
            # Add the completed test to the list. Make a copy of the 'test'
            # dictionary since we will re-use this object.
            tests.append(test.copy())

    # Append these results to the .csv file.
    if os.path.isfile('benchmark_tests.csv'):
        df = pd.read_csv('benchmark_tests.csv')
        df = df.append(pd.DataFrame(tests), ignore_index=True)
    else:
        df = pd.DataFrame(tests)
        
    # Write out the .csv file. 
    # I've specified a specific column order that I feel makes the data easier 
    # to read in a spreadsheet.
    df[['timestamp','gpu_name','n_gpu','metric','dataset_name','dataset_count',
        'vec_len','batch_size','k','latency','load_time','max_latency',
        'min_latency','repetitions','gpu_RAM','faiss_v',
        'instance']].to_csv('benchmark_tests.csv', index = False)
            