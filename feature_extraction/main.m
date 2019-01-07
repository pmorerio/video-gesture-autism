%

% Suppose a dataset structure with class folders in a root_dir and videos formatted as 'SS_XXX.ext (Subject_Trial.ext).

% You may need to install video plugins gstreamer0.10-ffmpeg
% gstreamer0.10-plugins gstreamer0.10-tools or similar

clear all; close all; clc;

%%%     CUSTOM SETTINGS 
root_dir = '/data/datasets/IIT_IFM_AUT/2D'; 
out_dir = '/data/datasets/IIT_IFM_AUT'; % must NOT be inside root_dir
ext = 'mp4';  % not .mp4
permute = true; % to mix the subjects  in the train/test/valid framework
split = [0.5, 0.25, 0.25]; % fraction of videos to be used for [train, test, validation]
crossval = true;
%

caffe_path ='/home/pmorerio/code/caffe';

use_gpu=1;
gpu_id = 0;
%
%%%

% Could be improved: the three steps are repeating some operations that
% could be performed globally here

% Step 1
% extract cnn features (GoogleNet) from all frames of all videos
% outputs a .mat file for each video
%cnn_feature_extraction(root_dir, out_dir, ext, caffe_path, use_gpu, gpu_id);

% % Step 2    
% % organize the dataset for training and testing 
% % two different schemes
% if crossval % 1-subject-out cross validation
     %create_txt_crossval(root_dir, out_dir, ext)
% else % split into train, test, valid
%     create_txt_lists(root_dir, out_dir, ext, split, permute);
% end
% % 
% % 
% % % Step 3
% % % Stuff all the feature in a single h5 file
% % % Make sure the files do not exist already
% if crossval % 1-subject-out cross validation
     h5_encode_crossval(root_dir, out_dir);
% else % split into train, test, valid
%     h5_encode(out_dir);
% end
