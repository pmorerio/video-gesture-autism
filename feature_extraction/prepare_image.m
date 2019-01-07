function batch = prepare_image(imbatch,IMAGE_MEAN)
    gauss_std=20;
    IMAGE_DIM = 256;
    CROPPED_DIM = 224; %no cropping actually....

    % resize to fixed input size
    cnt = 1;
    start = 0;
    % batch{cnt} = zeros(CROPPED_DIM, CROPPED_DIM, 3, 128, 'single');
    batch{cnt} = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single'); % to fit the bvcl implementation of googlenet
    for i=1:numel(imbatch)
    im = imbatch{i};
%     figure(1)
%     imshow(im)
    im=imgaussfilt(im, gauss_std); %gaussian smoothing 
%     figure(2)
%     imshow(im)
%     pause
    im = single(im);
    im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    % permute from RGB to BGR (IMAGE_MEAN is BGR)
    im = im(:,:,[3 2 1]) - IMAGE_MEAN;
    im = imresize(im, [CROPPED_DIM CROPPED_DIM], 'bilinear');
    batch{cnt}(:,:,:,i-start) = permute(im,[2,1,3]);
    if i-start==10
    % if i-start==128
        start = i;
        cnt = cnt+1;
        batch{cnt} = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
    % 	batch{cnt} = zeros(CROPPED_DIM, CROPPED_DIM, 3, 128, 'single');
    end
%
end
