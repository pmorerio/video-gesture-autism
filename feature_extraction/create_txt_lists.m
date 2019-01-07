% READS THE VIDEO DATASET AND SPLITS IT INTO TRAIN, TEST AND VALIDATION SETS
%OUTPUTS:
% xxxx_framenum.txt     list of the number of frames in each video
% xxxx_labels.txt       list of video labels
% xxxx_filename.txt     list of video filenames
function create_txt_lists(root_dir, out_dir, ext, split, permute)

if sum(split) ~= 1
    error('Impossible to split the dataset')
end

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

% open all output files
train_fileFRAME = fopen([out_dir '/train_framenum.txt'],'w');
train_fileNAME = fopen([out_dir '/train_filename.txt'],'w');
train_fileLAB = fopen([out_dir '/train_labels.txt'],'w');
val_fileFRAME = fopen([out_dir '/valid_framenum.txt'],'w');
val_fileNAME = fopen([out_dir '/valid_filename.txt'],'w');
val_fileLAB = fopen([out_dir '/valid_labels.txt'],'w');
test_fileFRAME = fopen([out_dir '/test_framenum.txt'],'w');
test_fileNAME = fopen([out_dir '/test_filename.txt'],'w');
test_fileLAB = fopen([out_dir '/test_labels.txt'],'w');
videos=dir([root_dir '/*mp4']);

train_n=floor( length(videos) * split(1));
test_n=floor( length(videos) * split(2));
valid_n=floor( length(videos) * split(3));
if(permute) idx=randperm(length(videos)); else idx=1:(length(videos)); end
for ff=1:train_n
    %          TRAIN
    ff
    fileName = videos(idx(ff)).name;
    fprintf(train_fileNAME,'%s \n',fileName);
    if fileName(5)=='C'
       fprintf(train_fileLAB,'%s \n','1');
    else
       fprintf(train_fileLAB,'%s \n','0');
    end
    readerobj = VideoReader([root_dir '/' fileName]);
    vidFrames = read(readerobj);
    fprintf(train_fileFRAME,'%s \n',num2str(size(vidFrames,4)));
end
for ff=(train_n+1):(train_n+test_n)
    ff
    %          TEST
    fileName = videos(idx(ff)).name;
    fprintf(test_fileNAME,'%s \n',fileName);
    if fileName(5)=='C'
       fprintf(test_fileLAB,'%s \n','1');
    else
       fprintf(test_fileLAB,'%s \n','0');
    end
    readerobj = VideoReader([root_dir '/' fileName]);
    vidFrames = read(readerobj);
    fprintf(test_fileFRAME,'%s \n',num2str(size(vidFrames,4)));
end
for ff=(train_n+test_n+1):(train_n+test_n+valid_n)
    ff
    %          VALIDATION
    fileName = videos(idx(ff)).name;
    fprintf(val_fileNAME,'%s \n',fileName);
    if fileName(5)=='C'
       fprintf(val_fileLAB,'%s \n','1');
    else
       fprintf(val_fileLAB,'%s \n','0');
    end
    readerobj = VideoReader([root_dir '/' fileName]);
    vidFrames = read(readerobj);
    fprintf(val_fileFRAME,'%s \n',num2str(size(vidFrames,4)));
end
end
%