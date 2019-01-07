% Extracts raw CNN features per frame
% It uses caffe bvlc_googlenet http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
% the prototxt file is cut to have the "inception_5b/output" as output
% layer (output feature cube dim = 7*7*1024)
function cnn_feature_extraction(root_dir, out_dir, ext, caffe_path, use_gpu, gpu_id)
crop_img=1130; %colonna dell'immagine fino a dove tenere
output_dir = [out_dir '/CNN_features_smooth_20'];
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

if exist([caffe_path '/matlab/+caffe'], 'dir')
    addpath([caffe_path '/matlab']);
else
    error('Please run this file from your matlab/+caffe dir');
end

model_dir = [caffe_path, '/models/bvlc_googlenet'];
model_def_file = [ model_dir '/deploy.prototxt_cut5b'];
model_file = [model_dir '/bvlc_googlenet.caffemodel'];


mean_file = [caffe_path '/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat'];
means = load(mean_file);

% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end

phase = 'test'; % run with phase test (so that dropout isn't applied)

% Initialize a network
net = caffe.Net(model_def_file, model_file, phase);

videos = dir([root_dir '/*.mp4']);


FeatDim = 7*7*1024; % output feature cube dim

for tt=1:length(videos)
    fileName = videos(tt).name;
    fprintf('Processing videos: video %d/%d,...\n',tt,length(videos));
    if exist([output_dir '/' fileName(1:end-length(ext)-1) '.mat'],'file')
        fprintf('[file exists...]');
        continue;
    end
    readerobj = VideoReader([ root_dir '/' fileName]);
    vidFrames = read(readerobj);
    imseq = cell(1,size(vidFrames,4));
    for j=1:size(vidFrames,4)
        imseq{j} =  vidFrames(:,1:crop_img,:,j); %croppo l'immagine per togliere eventuali pezzi di faccia a dx
    end
    
    % prepare oversampled input
    % input_data is Height x Width x Channel x Num
    input_data = prepare_image(imseq,means.mean_data);
    
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    feat = zeros(numel(imseq),FeatDim);
    cnt = 0;
    for bb=1:numel(input_data)
        scores = net.forward({input_data{bb}});
        %scores = caffe('forward', {input_data{i}});
        if cnt+10<=numel(imseq)
            feat(cnt+1:cnt+10,:)=reshape(scores{1},[FeatDim,10])';
            cnt = cnt + 10;
        else
            T = reshape(scores{1},[FeatDim,10])';
            feat(cnt+1:end,:) = T(1:numel(imseq)-cnt,:);
        end
    end
    feat = sparse(feat);
    save([output_dir '/' fileName(1:end-length(ext)) 'mat'],'feat', '-v7.3');
end

caffe.reset_all();
%
end
