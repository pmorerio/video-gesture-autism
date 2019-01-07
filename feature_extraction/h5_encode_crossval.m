function h5_encode_crossval(root_dir, out_dir)

    sets = {'train','test'}; %,'valid'};
    input_dir = [out_dir '/CNN_features_smooth_20']; % read from previously created .mat files
    if ~exist(input_dir,'dir')
        error('Cannot locate .mat features files');
    end
    
    dirinfo = dir(root_dir);
    dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
    dirinfo(ismember({dirinfo.name}, {'.', '..'})) = [];  %remove current and parent directory.

    classes = {dirinfo.name};
    
    % get the number of subjects in the dataset
%     subjinfo = dir([root_dir '/' classes{1}]);
%     subjinfo = {subjinfo.name};
%     subjinfo = subjinfo{end};
%     subjnum =  strsplit(subjinfo,'_');
%     subjnum = str2double(subjnum{1});
    subjnum = 40;

    for ss=1:subjnum
        %fprintf('%d\n',ss);
        for j=1:length(sets)
            [names] = textread([out_dir '/' sprintf('%.2d', ss) '_' sets{j} '_filename.txt'],'%s');
            if isempty(names)
                continue;
            end
            frames_list = dlmread([out_dir '/' sprintf('%.2d', ss) '_' sets{j} '_framenum.txt']);
            totalframes = sum(frames_list );

            file = [out_dir '/' sprintf('%.2d', ss) '_' sets{j} '_features.h5'];
            h5create(file, '/features', [50176 totalframes], 'Datatype', 'single', 'ChunkSize', [1024 size(frames_list ,1)]);
            h5disp(file);
            %matObj = matfile([out_dir '/' sprintf('%.2d', ss) '_' sets{j} ' _features.h5'],'Writable',true);
            %matObj.features(50176, totalframes) = single(0);

            frames_seen = 0;
            for i = 1:size(frames_list ,1)
                %fprintf('%d\n',i);
                name = strsplit(names{i},'/');
                name = strsplit(name{end},'.');
                name = name{1};
                filename = [input_dir,'/',name,'.mat']; 
                load(filename);
                feat = full(feat);
                feat = single(feat');
                %matObj.features(1:50176,frames_seen+1:frames_seen+frames(i)) = single(feat');
                h5write(file, '/features', feat, [1,frames_seen+1], size(feat));
                clear('feat');
                frames_seen = frames_seen + frames_list(i);
            end
        end
        
    end
%
end