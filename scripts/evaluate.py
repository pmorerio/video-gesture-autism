import numpy
import sys
import argparse
import time
import datetime

from src.actrec import train
from util.file_janitor import tidy_up

def main(job_id, params):
    
    print '-----------'   
    print 'Starting at', str(datetime.datetime.now())[:-7]
    print '-----------'
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
                                        dataset=params['dataset'][0], 
                                        dictionary=params['dictionary'][0],
                                        use_dropout=params['use_dropout'][0],
                                        reload_=params['reload'][0],
                                        training_stride=params['training_stride'][0],
                                        testing_stride=params['testing_stride'][0],
                                        last_n=params['last_n'][0],
                                        fps=params['fps'][0],
                                        data_dir=params['data_dir'][0]
                             )
    
    train_time = time.time() - train_start
    print "Execution time:", train_time
    print "Training accuracy", trainacc
    print "Validation accuracy", validacc
    print "Test accuracy", testacc
    
    res_file = open('res.txt','w')
    print>>res_file, "Execution time:", train_time, '\n'
    print>>res_file, "Training accuracy", trainacc, '\n'
    print>>res_file, "Validation accuracy", validacc, '\n'
    print>>res_file, "Test accuracy", testacc, '\n'
    res_file.close()
        
    # move output files to /old dir
    tidy_up()
    return validacc

if __name__ == '__main__':
    options = {
        'dim_out': [16],		# hidden layer dim for outputs
        'ctx_dim': [1024],		# context vector dimensionality
        'dim': [64],			# the number of LSTM units
        'n_actions': [40],		# number of classes to predict
        'n_layers_att':[1],
        'n_layers_out': [1],
        'n_layers_init': [1],
        'ctx2out': [False],
        'patience': [10],
        'max_epochs': [10],
        'dispFreq': [100],
        'decay_c': [0.0001],        	# gammma in eq (7)
        'alpha_c': [0.0],           	# lambda in eq (7) (?)
        'temperature_inverse': [1],
        'learning_rate': [0.005],
        'selector': [False],
        'maxlen': [15],                  # looks like this is actually the length of the chunk of video used as sample 
        'optimizer': ['adam'],
        'batch_size': [64],
        'valid_batch_size': [128],
        'model': ['model_aut.npz'],
        'validFreq': [200000],             	# should be smaller than the num of training batches to have at least one extra validation within each epoch
        'saveFreq': [200000],		# save the parameters after every saveFreq updates 
        'dataset': ['IfM'],
        'dictionary': [None],
        'use_dropout': [True],
        'reload': [False],
        'training_stride': [1],
        'testing_stride': [1],
        'last_n': [15],			# timesteps from the end used for computing prediction
        'fps': [100],
        'data_dir':['/data/datasets/IIT_IFM_AUT/split_30_subj_train_CNN_frame_smooth_30_class3/']
    }

    if len(sys.argv) > 1:
        options.update(eval("{%s}"%sys.argv[1]))

    main(0, options)


