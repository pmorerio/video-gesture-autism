import numpy
import sys
import argparse
import time

from src.actrec import train
from util.file_janitor import tidy_up

import os
import glob

def main(job_id, params):
    
    dataset_dir = params['data_dir'][0]
    # how many subjects do we have? (trying to keep everything general)
    num_subj = len(glob.glob(dataset_dir + '/*_train_filename.txt'))
    
    # prepare a container where to stuff (append) accuracies
    history_train = []
    history_valid = []
    history_test = []
    history_time = []
    
    
    for subj in range(num_subj):
        
        subj_out = '%02d' % (subj+1)		# suppose less than 99 subjects...
        
        # check for any missing subject in the datset
        if not os.path.exists(dataset_dir + subj_out + '_test_features.h5'):
			continue
        
        #dataset_name = 'IfM_' + subj_out
        dataset_name = 'IfM_' + subj_out
	print '---------------------------------------------------------------------'
	print 'Now evaluating: subject', subj_out
        print 'Anything printed here will end up in the output directory for job #%d' % job_id
        print params
	
	train_start = time.time()
        trainacc, validacc, testacc = train(dim_out=params['dim_out'][0],
					    ctx_dim=params['ctx_dim'][0],
					    dim=params['dim'][0],
					    n_actions=params['n_actions'][0],
					    n_layers_att=params['n_layers_att'][0],
					    n_layers_out=params['n_layers_out'][0],
					    n_layers_init=params['n_layers_init'][0],
					    ctx2out=params['ctx2out'][0],
					    patience=params['patience'][0],
					    max_epochs=params['max_epochs'][0],
					    dispFreq=params['dispFreq'][0],
					    decay_c=params['decay_c'][0],
					    alpha_c=params['alpha_c'][0],
					    temperature_inverse=params['temperature_inverse'][0],
					    lrate=params['learning_rate'][0],
					    selector=params['selector'][0],
					    maxlen=params['maxlen'][0],
					    optimizer=params['optimizer'][0], 
					    batch_size=params['batch_size'][0],
					    valid_batch_size=params['valid_batch_size'][0],
					    saveto=params['model'][0],
					    validFreq=params['validFreq'][0],
					    saveFreq=params['saveFreq'][0],
					    #dataset=params['dataset'][0], 
					    dataset=dataset_name,
					    dictionary=params['dictionary'][0],
					    use_dropout=params['use_dropout'][0],
					    reload_=params['reload'][0],
					    training_stride=params['training_stride'][0],
					    testing_stride=params['testing_stride'][0],
					    last_n=params['last_n'][0],
					    fps=params['fps'][0],
					    data_dir=params['data_dir'][0])
								
								    
	train_time = time.time() - train_start
	print "Execution time:", train_time
	print "Training accuracy", trainacc
	print "Validation accuracy", validacc
	print "Test accuracy", testacc
	
	# save history before moving to a new subj
	history_train.append(trainacc)
        history_valid.append(validacc)
        history_test.append(testacc)
        history_time.append(train_time)
									
	#save in a file history up to now
	train_file = open('train_acc.txt','w')
	valid_file = open('valid_acc.txt','w')
	test_file = open('test_acc.txt','w')
	time_file = open('times.txt','w')
	
	for item in history_train:
	    print>>train_file, item 
	for item in history_valid:
	    print>>valid_file, item 
	for item in history_test:
	    print>>test_file, item 
	for item in history_time:
	    print>>time_file, item
	
	train_file.close()
	valid_file.close()
	test_file .close()
	time_file.close()
	
	# move all files to /old before moving on
        tidy_up(str(subj_out))
	# we can finally move on to the next subject...
	
    # this shouldn't be need, but just in case
    tidy_up()
    
    return 1

if __name__ == '__main__':
    options = {
        'dim_out': [16],		# hidden layer dim for outputs
        'ctx_dim': [1024],		# context vector dimensionality
        'dim': [128],			# the number of LSTM units
        'n_actions': [2],		# number of digits to predict
        'n_layers_att':[1],
        'n_layers_out': [1],
        'n_layers_init': [1],
        'ctx2out': [False],
        'patience': [10],
        'max_epochs': [2],
        'dispFreq': [100],
        'decay_c': [0.0001],       	# gammma in eq (7)
        'alpha_c': [0.],           	# lambda in eq (7) (?) (changed to ATTENTION FOCUS param in my implementation)
        'temperature_inverse': [1],
        'learning_rate': [0.00005],
        'selector': [False],
        'maxlen': [10],                  # looks like this is actually the length of the video sample 
        'optimizer': ['adam'],
        'batch_size': [64],
        'valid_batch_size': [128],
        'model': ['model.npz'],
        'validFreq': [30000000],             	# should be smaller than the num of training batches to have at least one extra validation within each epoch
        'saveFreq': [30000000],		# save the parameters after every saveFreq updates 
        'dataset': ['IfM'],		# overwritten to 'IfM_XX' in the main 
        'dictionary': [None],
        'use_dropout': [True],
        'reload': [False],
        'training_stride': [1],
        'testing_stride': [1],
        'last_n': [10],			# timesteps from the end used for computing prediction
        'fps': [100],
	'data_dir':['/data/datasets/IIT_IFM_AUT/full_data/']
    }

    if len(sys.argv) > 1:
        options.update(eval("{%s}"%sys.argv[1]))

    main(0, options)


