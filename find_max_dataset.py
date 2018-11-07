'''
This script is used to determine, through experimentation, the maximum dataset
size supported by FAISS on a particular GPU configuration.

This script attempts to load a dataset of a particular size, and records 
whether the load succeeded or failed. All of the attempts (and whether each 
failed or succeeded) are stored in a .csv file: `max_dataset_size.csv`

On each attempt, it looks at the largest size that has succeeded and the 
smallest size that has failed, and then tries the size halfway between them.

The script rounds the vector counts to the nearest 10,000 so that the numbers
are a little less ugly. But you can remove this step if you wish to find the 
*exact* number.

One unfortunate detail is that when the dataset is too large, cudaMalloc throws
an exception which crashes Python, so we can't programmatically determine that
the size was too large. To work around this, the script first records
the test run as a _failure_ in the .csv table, and then goes back and changes 
it to a success if it doesn't crash.
'''
from __future__ import division

import time
import sys
import numpy as np
import faiss
import pandas as pd

# ==============================================
#        Server Properties
# ==============================================


test = {# Specify properties of the GPU
        'gpu_name': "Tesla K80",
        'n_gpu': 8,
        'gpu_RAM': 8 * 12,
        'faiss_v': '1.4.0', 
         
        # Specify properties of the dataset
        'vec_len': 300,
        'success': False
    }

# ==============================================
#        Identify Next Dataset Size to Try
# ==============================================

print('\nSearching for Maximum Dataset Size with:')
print('    %dx %s with %d GB RAM\n' % (test['n_gpu'], test['gpu_name'], int(test['gpu_RAM'])))
 
# Calculate the theoretical maximum dataset vector count based on the RAM.
theoret_max = int((test['gpu_RAM'] * 2**30) / (test['vec_len'] * 4))

# Select the trials matching this test's parameters.
df = pd.read_csv('max_dataset_size.csv')
df = df.loc[(df.gpu_name == test['gpu_name']) &
            (df.n_gpu == test['n_gpu']) &
            (df.gpu_RAM == test['gpu_RAM']) &
            (df.faiss_v == test['faiss_v']) &
            (df.vec_len == test['vec_len'])]

# If there are no successful trials yet, then start with 70% of the theoretical max.
if len(df.loc[df.success == True]) == 0:
    max_success = int(0.7 * theoret_max)
# Otherwise, use the maximum successful size.
else:   
    max_success = int(df.loc[df.success == True].vec_count.max())

# If there are no failures on record, start with 90% of the theoretical max.
if len(df.loc[df.success == False]) == 0:
    min_fail = int(0.9 * theoret_max)
# Otherwise, use the minimum failed size
else:    
    min_fail = int(df.loc[df.success == False].vec_count.min())
  
# Calculate the next size to try.
try_size = int((min_fail - max_success) / 2) + max_success

# Round it to the nearest 10,000.
try_size = int(round(try_size / 10000.0) * 10000.0)

print('Theoret Max: %s' % "{:,}".format(theoret_max))
print('Max Success: %s' % "{:,}".format(max_success))
print('Min Failure: %s' % "{:,}".format(min_fail))
print('Next To Try: %s' % "{:,}".format(try_size))

if (try_size == min_fail) or (try_size == max_success):
    print('We have converged!')
    sys.exit(0)

test['vec_count'] = try_size   

# =====================================
#       Load or Generate Dataset
# =====================================

# Record the name of the dataset.
generate = False

t0 = time.time()

if generate:
    print('Generating dataset...')
    sys.stdout.flush()
            
    vecs = np.random.rand(test['vec_count'], test['vec_len']).astype('float32')

else:
    print('Loading dataset...')
    sys.stdout.flush()
    
    # Open a memory mapped version of the matrix.
    vecs_mmap = np.load('./%d_x_%d.npy' % (int(130E6), int(300)), mmap_mode='r')
    
    # Read the portion of the dataset that we need.
    vecs = vecs_mmap[0:test['vec_count'],:]
    
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

# Record the test in the table. 
# If the test crashes during the malloc below, then the failure will stand.
# If the test succeeds, then at the end of this script we go back and fix the
# result.
df = pd.read_csv('max_dataset_size.csv')
df = df.append(pd.DataFrame([test]), ignore_index=True)
df.to_csv('max_dataset_size.csv', index = False)

# Add vecs to our GPU index
try:
    print('Adding dataset to index...')
    sys.stdout.flush()

    t0 = time.time()    
    
    gpu_index.add(vecs)
    
    print('Building index took %.2f seconds' % (time.time() - t0))
    
    # ======================================
    #          Attempt to Query
    # ======================================
    
    print('Attempting batch query of 1,024 vectors with k=100...')
    t0 = time.time()    
    
    # Attempt a big batch query to ensure it doesn't crash on this.
    D, I = gpu_index.search(vecs[:1024], k=100) 
    print('    Done. Batch query took %.2f seconds' % (time.time() - t0))
    
# The cudaMalloc error can't be caught, unfortunately...    
except:
    print("Error:", sys.exc_info()[0])
    raise
    
# ======================================
#         Record Result
# ======================================

# If we made it here, the test passed, so set success = True.
df.at[len(df) - 1, 'success'] = True
df.to_csv('max_dataset_size.csv', index = False)
