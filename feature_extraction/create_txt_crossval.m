% READS THE VIDEO DATASET AND ITERATIVELY SPLITS IT INTO TRAIN, TEST AND VALIDATION SETS
% ONE-SUBJECT-OUT CROSS VALIDATION (ONE OUT FOR VALIDATION, ONE OUT FOR TESTING)
%OUTPUTS (SS is the subject out for testing, one random is out for validation):
        % SS_xxxx_framenum.txt     list of the number of frames in each video
        % SS_xxxx_labels.txt       list of video labels
        % SS_xxxx_filename.txt     list of video filenames     
function create_txt_crossval(root_dir, out_dir, ext)

    dirinfo = dir(root_dir);
    dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
    dirinfo(ismember({dirinfo.name}, {'.', '..'})) = [];  %remove current and parent directory.

    classes = {dirinfo.name};
    
    if ~exist(out_dir,'dir')
        mkdir(out_dir);
    end

    % get the number of subjects in the dataset
%     subjinfo = dir([root_dir]);
%     subjinfo = {subjinfo.name};
%     subjinfo = subjinfo{end};
%     subjnum =  strsplit(subjinfo,'_');
%     subjnum = str2double(subjnum{1});

    subjnum = 40;
    
    for ss=1:subjnum % not a very smart implementation: I am reading video files $subjnum times
        ss

        % open all otput files
        train_fileFRAME = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'train_framenum.txt'],'w');
        train_fileNAME = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'train_filename.txt'],'w');
        train_fileLAB = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'train_labels.txt'],'w');
%         val_fileFRAME = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'valid_framenum.txt'],'w');
%         val_fileNAME = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'valid_filename.txt'],'w');
%         val_fileLAB = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'valid_labels.txt'],'w');
        test_fileFRAME = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'test_framenum.txt'],'w'); 
        test_fileNAME = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'test_filename.txt'],'w');
        test_fileLAB = fopen([out_dir '/' sprintf('%.2d', ss) '_' 'test_labels.txt'],'w');
        
        % validation subject is the following for simplicity
%         val_subj = mod(ss, subjnum) + 1;

        for cc=1:length(classes)
            %class_videos=dir([root_dir '/' classes{cc} '/*' ext]);
            test_vids = dir([root_dir '/' classes{cc} '/S' sprintf('%.3d', ss) '_*' ext]);
            label_test = 0; 
            if isempty({test_vids.name})
                test_vids = dir([root_dir '/' classes{cc} '/S' sprintf('%.3d', ss-20) 'C*' ext]);
                label_test=1;
            end
%             valid_vids = dir([root_dir '/' classes{cc} '/' sprintf('%.2d', val_subj) '*' ext]);
%             while isempty({valid_vids.name})
%                 valid_vids = dir([root_dir '/' classes{cc} '/' sprintf('%.2d', val_subj+1) '*' ext]);
%             end
            train_vids = dir([root_dir '/' classes{cc} '/*' ext]);
            train_vids(ismember({train_vids.name}, {test_vids.name})) = [];
%             train_vids(ismember({train_vids.name}, {valid_vids.name})) = [];
             
            for ff=1:length(train_vids)
                %          TRAIN
                fileName = train_vids(ff).name;
                fprintf(train_fileNAME,'%s \n',[classes{cc} '/' fileName]);
                label = length(strsplit(fileName, 'C')) -1;
                fprintf(train_fileLAB,'%s \n',num2str(label));
                readerobj = VideoReader([root_dir '/' classes{cc} '/' fileName]);
                vidFrames = read(readerobj);
                fprintf(train_fileFRAME,'%s \n',num2str(size(vidFrames,4)));
            end
            for ff=1:length(test_vids)
                %          TEST
                fileName = test_vids(ff).name;
                fprintf(test_fileNAME,'%s \n',[classes{cc} '/' fileName]);
                fprintf(test_fileLAB,'%s \n',num2str(label_test));
                readerobj = VideoReader([root_dir '/' classes{cc} '/' fileName]);
                vidFrames = read(readerobj);
                fprintf(test_fileFRAME,'%s \n',num2str(size(vidFrames,4)));
            end
%             for ff=1:length(valid_vids)
%                 %          VALIDATION
%                 fileName = valid_vids(ff).name;
%                 fprintf(val_fileNAME,'%s \n',[classes{cc} '/' fileName]);
%                 fprintf(val_fileLAB,'%s \n',num2str(cc-1));
%                 readerobj = VideoReader([root_dir '/' classes{cc} '/' fileName]);
%                 vidFrames = read(readerobj);
%                 fprintf(val_fileFRAME,'%s \n',num2str(size(vidFrames,4)));
%             end 
        end
    end
%    
end