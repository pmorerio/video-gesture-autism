## Analyzes the accuray contribution of different portions of the videos
from __future__ import print_function
import numpy
from scipy import stats


num_subj = 40
SPLITS = 4

for subj in range(num_subj):
        
	subj_out = '%02d' % (subj+1)	

	tempfilename ='/home/pmorerio/code/Intention-from-Motion/old/smooth_20_full_data_dim64/' + subj_out + '/test_results_last15_model.txt'
	labels_file='/data/datasets/IIT_IFM_AUT/full_data/' + subj_out + '_test_labels.txt'
	num_frames_file='/data/datasets/IIT_IFM_AUT/full_data/' + subj_out + '_test_framenum.txt'

	f = open(tempfilename,'r')
	lines = f.readlines()
	f.close()

	pred  = numpy.zeros( (len(lines), SPLITS+1) ).astype('int64')
	#pred  = numpy.zeros( len(lines) ).astype('int64')
	for i in xrange(len(lines)):
		try:
			s=lines[i].split(' ')[1]
			s=s[0:-1]
			s=s.split(',')
			# s is the whole sequence. 
			s = [int(x) for x in s]
			s = numpy.array(s)
			# approx split in $SPLITS parts. 
			spl = numpy.split(s,  [(ii+1)*len(s)/SPLITS for ii in range(SPLITS)])[:-1]
			for ii, elem in enumerate(spl):
				spl[ii] = stats.mode(elem)[0][0]

			# TODO: concatenate also (to have progressive accuracy)
			# ...
			s = stats.mode(s)[0][0] #prediction for video i is the MODE of the sequence of labels
			pred[i][0] = int(s)
			pred[i][1:] = [int(elem) for elem in spl]
		except IndexError:
			print('One blank index skipped')
			pred[i][:] = -1


	f = open(labels_file,'r')
	lines = f.readlines()
	f.close()
	#f = open(num_frames_file,'r')
	#framenum = f.readlines()
	#f.close()

	truth  = numpy.zeros(len(lines)).astype('int64')
	framel = numpy.zeros(len(lines)).astype('int64')
	for i in xrange(len(lines)):
		s=lines[i][0:-1]
		truth[i] = int(s)
		#framel[i]= int(framenum[i][0:-1])
	
	#print [(truth==pred[:,kk]).mean() for kk in range(SPLITS+1)]
	for kk in range(SPLITS+1):
		print( (truth==pred[:,kk]).mean(), end="\t" )
	print('')
