## TODO Jan 12 2016
#       - save in results folder
#       - superimpose PREDICTED LABEL (and/or wrong/correct prediction) reading it from *results.txt file
#           or from f_preds. This should also help understanding what happens when classification fails
#       - statistics for the attention map
#       - cycle over subjects (only on test set?)

import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pkl
import numpy
import scipy
import os

#import matplotlib.pyplot as plt

import cv2
import skimage
import skimage.transform
import skimage.io
from PIL import Image

import sys
sys.path.append('../')

from util.data_handler import DataHandler
from util.data_handler import TrainProto
from util.data_handler import TestTrainProto
from util.data_handler import TestValidProto
from util.data_handler import TestTestProto

import src.actrec

def overlay(bg,fg):
    """
    Overlay attention over the video frame
    """
    src_rgb = fg[..., :3].astype(numpy.float32) / 255.
    src_alpha = fg[..., 3].astype(numpy.float32) / 255.
    dst_rgb = bg[..., :3].astype(numpy.float32) / 255.
    dst_alpha = bg[..., 3].astype(numpy.float32) / 255.

    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] + dst_rgb * dst_alpha[..., None] * (1. - src_alpha[..., None])) / out_alpha[..., None]

    out = numpy.zeros_like(bg)
    out[..., :3] = out_rgb * 255
    out[..., 3] = out_alpha * 255

    return out
    
def add_alphalayer(image, alpha, concatenate=True):    
    """
    Returns a numpy array with original image + overalayed
    """
    # alpha is here a 49-dim vector
    image = cv2.resize(image, (224, 224))
    img = numpy.array(image)
    alphalayer = numpy.ones((224,224,1))*255
    img = numpy.dstack((img,alphalayer)) #rgba

    # create the attention map and add an Alpha layer to the RGB image
    # 7*32 = 224 must upscale to original image 
    alpha_img = skimage.transform.pyramid_expand(alpha.reshape(7,7), upscale=32, sigma=20)
    alpha_img = alpha_img*255.0/numpy.max(alpha_img)
    alpha_img = skimage.color.gray2rgb(alpha_img)
    alpha_img = numpy.dstack((alpha_img,0.8*alphalayer)) #rgba

    old_img = img
    img = overlay(img,alpha_img)
    if concatenate:
        img  = numpy.concatenate((old_img,img),axis=1) # axis=0) 
    return img


num_subj = 40

for subj in range(num_subj):
        
	subj_out = '%02d' % (subj+1)	

model ='/home/pmorerio/code/Intention-from-Motion/old/smooth_20_full_data_dim64/01/model.npz'
# PROBLEM here in videopath: subfolders
videopath = '/data/datasets/IIT_IFM_AUT/2D/'
dataset ='IfM_01'
tbidx = 5 # (tbidx+1)-th video in the test_filename,txt file

with open('%s.pkl'%model, 'rb') as f:
    options = pkl.load(f)

batch_size = 1
data_dir='/data/datasets/IIT_IFM_AUT/full_data/'
# try out different fps for visualization
fps = options['fps']
#fps = 100   # to see further on along the sequence
#skip = int(100/fps)


flen =[]
for line in open(data_dir+'01_test_framenum.txt'): # test
    flen.append(line.strip())
    
    
maxlen = int(flen[tbidx]) # to get the alphas for the whole tbidx-th video
print 'Video length:', maxlen


print '-----'
#print 'Skip set at', skip
print 'Booting up the data handler'

data_pb = TestTestProto(batch_size,maxlen,maxlen,dataset,data_dir, fps) # or TestTrainProto or TestValidProto
dh = DataHandler(data_pb)
dataset_size = dh.GetDatasetSize()
num_batches = dataset_size / batch_size

print 'Data handler ready'
print '-----'
params  = src.actrec.init_params(options)
params  = src.actrec.load_params(model, params)
tparams = src.actrec.init_tparams(params)

trng, use_noise, inps, alphas, cost, opt_outs, preds = src.actrec.build_model(tparams, options)
f_alpha = theano.function(inps,alphas,name='f_alpha',on_unused_input='ignore')
#f_preds = theano.function(inps,preds,profile=False,on_unused_input='ignore')

mask = numpy.ones((maxlen, batch_size)).astype('float32')

x, y, fname = dh.GetSingleExample(data_pb,tbidx)
alpha = f_alpha(x,mask,y)
print 'Attention map (alpha):', alpha.shape
print 'Reading from', videopath+fname
out_folder = videopath+'att_'+fname[:-4]
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    print 'Creating', out_folder, '...'
vidcap = cv2.VideoCapture(videopath+fname)
#space = 255.0*numpy.ones((224*2,20,4))
#space[:,:,0:3] = 255.0*numpy.ones((224*2,20,3))

#imgf = numpy.array([]).reshape(2*224,0,4)    
res_file = open(out_folder+'/res.txt','w')

for ii in xrange(alpha.shape[0]): 
    print>>res_file, [alpha[ii,0,jj] for jj in range(49)]
    success, current_frame = vidcap.read()
    if not success:
        break
        
    # add an Alpha layer to the RGB image
    img = add_alphalayer(current_frame, alpha[ii,0,:])
    
    save2 = out_folder+'/'+'%06d' % ii + '.png'
    cv2.imwrite(save2,img)
    
res_file.close()
