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
        'n_gpu': 8,
        'gpu_RAM': 8 * 12,
        'faiss_v': '1.4.0', 
        'instance': 'p2.8xlarge',
        
        # Specify some test properties
        'metric': 'L2'
    }

# These are all of the columns in our output. This array holds column order.
column_names = ['timestamp','gpu_name','n_gpu','metric',
                 'dataset_name','dataset_count', 'vec_len', 'dtype', 
                 'batch_size','k',
                 'latency','load_time','max_latency','min_latency',
                 'repetitions','gpu_RAM','faiss_v','instance','notes']

# TODO 
#  DONE - 1x VSX Max
#  DONE - 1x V100_f32 Max
#  DONE - 1x V100_f16 Max
#  DONE - 4x VSX Max
#  DONE - 4x V100 Max f32 and f16
#  DONE - 10x VSX Max
#  DONE - 8x V100 f32 and f16 Max
#  DONE - 16x K80 Max                 

dataset_sizes = [                 
                 (     int(1e6), 96, '1M x 96'),
                 (    int(10e6), 96, '10M x 96'),
                 ( int(25.07e6), 96, '1x K80 Max Capacity'),
                 ( int(33.43e6), 96, '1x V100 f32 Max Capacity'), # Max TBD
                 (    int(50e6), 96, '50M x 96'),
                 ( int(50.15e6), 96, '2x K80 Max Capacity'), 
                 ( int(66.86e6), 96, '1x V100 f16 Max Capacity'), # Max TBD
                 (   int(100e6), 96, '100M x 96'),
                 (  int(133.69), 96, '4x V100 f32 Max Capacity'), # Max TBD
                 (   int(200e6), 96, '200M x 96'),
                 (int(200.54e6), 96, '8x K80 Max Capacity'), 
                 (int(267.38e6), 96, '8x V100 f32 Max Capacity'), # Max TBD
                 (int(267.38e6), 96, '4x V100 f16 Max Capacity'), # Max TBD
                 (   int(400e6), 96, '400M x 96'), 
                 (int(401.08e6), 96, '16x K80 Max Capacity'), 
                 (int(469.76e6), 96, '1x VSX Max Capacity'), # 469.762048e6
                 (int(534.76e6), 96, '8x V100 f16 Max Capacity'), # Max TBD
                 ( int(1.879e9), 96, '4x VSX Max Capacity'), # 1.879048192e9
                 ( int(4.697e9), 96, '10x VSX Max Capacity'), # 4.69762048e9
                  
                 (     int(1e6), 300, '1M x 300'),
                 (     int(5e6), 300, '5M x 300'),
                 (  int(8.08e6), 300, '1x K80 Max Capacity'),
                 (    int(10e6), 300, '10M x 300'),
                 ( int(10.77e6), 300, '1x V100 f32 Max Capacity'), # Max TBD
                 ( int(21.54e6), 300, '1x V100 f16 Max Capacity'), # Max TBD
                 ( int(43.08e6), 300, '4x V100 f32 Max Capacity'), # Max TBD
                 (    int(50e6), 300, '50M x 300'),
                 ( int(64.56e6), 300, '8x K80 Max Capacity'),
                 ( int(86.16e6), 300, '4x V100 f16 Max Capacity'), # Max TBD
                 ( int(86.16e6), 300, '8x V100 f32 Max Capacity'), # Max TBD
                 (int(129.12e6), 300, '16x K80 Max Capacity'),
                 (int(150.32e6), 300, '1x VSX Max Capacity'), # 150.323855e6
                 (int(172.32e6), 300, '8x V100 f16 Max Capacity'), # Max TBD
                 (int(601.28e6), 300, '4x VSX Max Capacity'), # ...                 
                 ( int(1.503e9), 300, '10x VSX Max Capacity'), # 1.5032385536e9
                 
                 (   int(100e3), 4096, '100K x 4096'),
                 (   int(590e3), 4096, '1x K80 Max Capacity'),
                 (   int(780e3), 4096, '1x V100 f32 Max Capacity'),
                 (     int(1e6), 4096, '1M x 4096'),
                 (  int(1.56e6), 4096, '1x V100 f16 Max Capacity'),
                 (     int(2e6), 4096, '2M x 4096'),
                 (  int(3.12e6), 4096, '4x V100 f32 Max Capacity'), # Max TBD
                 (  int(6.24e6), 4096, '4x V100 f16 Max Capacity'), # Max TBD
                 (  int(6.24e6), 4096, '8x V100 f32 Max Capacity'), # Max TBD
                 (  int(4.74e6), 4096, '8x K80 Max Capacity'), 
                 (  int(9.48e6), 4096, '16x K80 Max Capacity'), 
                 ( int(11.01e6), 4096, '1x VSX Max Capacity'), # 11.010048e6
                 ( int(12.28e6), 4096, '8x V100 f16 Max Capacity'), # Max TBD
                 ( int(44.04e6), 4096, '4x VSX Max Capacity'),
                 ( int(110.1e6), 4096, '10x VSX Max Capacity'), # 110.10048e6
                 
                ]


def get_max_capacity(test):
    '''
    This function looks up the maximum number of vectors that the current GPU
    configuration can support at this vector length.
    '''
    
    # Select the trials matching this test's parameters.
    df = pd.read_csv('max_dataset_size.csv')
    df = df.loc[(df.gpu_name == test['gpu_name']) &
                (df.n_gpu == test['n_gpu']) &
                (df.gpu_RAM == test['gpu_RAM']) &
                (df.faiss_v == test['faiss_v']) &
                (df.vec_len == test['vec_len'])]
    
    # Error if there's no matching dataset size experiments.
    if df.empty or df.loc[df.success == True].empty:
        print('ERROR - No capacity data for this configuration')
        return -1
    
    # Find the maximum successful size and the minimum fail size.    
    max_success = int(df.loc[df.success == True].vec_count.max())
    min_fail = int(df.loc[df.success == False].vec_count.min())
    
    # Warn if the capacity experiments haven't converged yet.
    if (min_fail - max_success) > 10000:
        print('WARNING: Capacity experiments not complete for %dx %s with length %d' % (test['n_gpu'], test['gpu_name'], test['vec_len']))
    
    return max_success
   
    

batch_sizes = [1, 16, 64, 256, 1024]
ks = [1, 10, 100]
repetitions = 10

# Memory map the dataset files. We'll only load the portions that we need 
# for each experiment into memory
vecs_mmap = {96: np.load('210000000_x_96.npy', mmap_mode='r'),
             300: np.load('100000000_x_300.npy', mmap_mode='r'),
             4096: np.load('5000000_x_4096.npy', mmap_mode='r')}

# For each dataset size...
for ds_shape in dataset_sizes:

    print('===================================================')
    print('Running benchmark with dataset %s x %d' % ("{:,}".format(ds_shape[0]), ds_shape[1]))
    print('===================================================')
    
    # Create a test object for storing all parameters and results of this experiment
    test['dataset_count'] = ds_shape[0]
    test['vec_len'] = ds_shape[1]
    
    # Lookup the maximum supported capacity for the current GPU configuration.
    max_count = get_max_capacity(test)
    
    # Check if we can support this size.
    if test['dataset_count'] > max_count:
        print('Dataset size not supported, skipping.')
        continue
    
    # =====================================
    #          Generate Dataset
    # =====================================

    # Record the name of the dataset.
    test['dataset_name'] = 'random'
    test['notes'] = ds_shape[2]
    
    generate = False

    t0 = time.time()
    
    if generate:
        print('Generating dataset...')
        sys.stdout.flush()
                
        vecs = np.random.rand(ds_shape[0], ds_shape[1]).astype('float32')
    
    else:
        print('Loading dataset...')
        sys.stdout.flush()
        
        # Read the portion of the dataset that we need.
        vecs = vecs_mmap[ds_shape[1]][0:ds_shape[0],:]
        
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
    
    # FAISS uses 32-bit floats.
    test['dtype'] = 'float32'
    
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
    df[column_names].to_csv('benchmark_tests.csv', index = False, float_format='%.6f')
            