function h5_encode(out_dir)

    sets = {'train','test','valid'};

    input_dir = [out_dir '/CNN_features_smooth_30']; % read from previously created .mat files

    if ~exist(input_dir,'dir')
        error('Cannot locate .mat features files');
    end
    
    
    for j=1:length(sets)
        sets{j}
        [names] = textread([out_dir '/' sets{j} '_filename.txt'],'%s');
        frames_list = dlmread([out_dir '/' sets{j} '_framenum.txt']);
        totalframes = sum(frames_list );

        file = [out_dir '/' sets{j} '_features.h5'];
        h5create(file, '/features', [50176 totalframes], 'Datatype', 'single', 'ChunkSize', [1024 size(frames_list ,1)]);
        h5disp(file);
        %matObj = matfile([out_dir '/' sets{j} '_features.h5'],'Writable',true);
        %matObj.features(50176, totalframes) = single(0);

        frames_seen = 0;
        for i = 1:size(frames_list ,1)
            fprintf('%d\n',i);
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
%
end
