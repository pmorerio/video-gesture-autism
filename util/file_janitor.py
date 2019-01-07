import os
import shutil
import datetime
import glob

def tidy_up(suffix=''):
	
	# check for old model files
	dirty_dir = len(glob.glob('*.npz'))
	
	if dirty_dir:
		print 'Cleaning up old models...'
		now = datetime.datetime.now()
		if len(suffix)==0:
			dest_dir = 'old/'+ str(now)[:-7]
		else:	
			dest_dir = 'old/'+ suffix #get rid of milliseconds + add subject number if crossval
		
		if not os.path.exists(dest_dir):
			os.makedirs(dest_dir)
		
		for file in glob.glob('*.txt'):
			print 'Moving', file                                                                                                                                        
			shutil.move(file, dest_dir)
			
		for file in glob.glob('*.pkl'):
			print 'Moving', file                                                                                                                                        
			shutil.move(file, dest_dir)	
			
		for file in glob.glob('*.npz'):
			print 'Moving', file                                                                                                                                        
			shutil.move(file, dest_dir)	

	return 0
