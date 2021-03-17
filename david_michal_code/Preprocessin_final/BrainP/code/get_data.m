function run_calc( path, folder_name )
    num_channels = 8;
    num_samples = 32;
    global sDavid;
    C = strsplit(path,'/');
    delim = '/';
    if size(C,2)==1
        C = strsplit(path,'\');
        delim = '\';
    end
    if path(end) ~= delim
        path = strcat(path,delim);
    end
    file_prefix = C(end);
    file_prefix = folder_name;
    [valid_vecs] =  get_valid_vecs(strcat(path));
    num_shanks = max(sDavid.shankclu(:,1));

%% save mean

    for shank = 1:num_shanks
        spk_path = char(strcat(path,file_prefix,'.spk.',num2str(shank)));
        try
            spk = readspk(spk_path,num_channels, num_samples,[] , 'int16', 'single');
        catch
            disp(['failed on ',file_prefix]);
        end 
        disp(spk_path); 
        clu_path = char(char(strcat(path,file_prefix,'.clu.',num2str(shank))));
        clu = importdata(clu_path);
        clu_for_day = clu(2:end);
        disp(clu_path);   
        res_path = char(strcat(path,file_prefix,'.res.',num2str(shank)));
        res = importdata(res_path);
        disp(res_path); 


        idx1 = strcmp( sDavid.filename, folder_name );
        idx2 = ismember( sDavid.shankclu( :, 1), shank );
        idx = idx1 .* idx2;
        group_for_shank = sDavid.shankclu(idx~=0,2);

        for group=1:length(group_for_shank)
        disp([shank group_for_shank(group)]);
    %     if(group_for_shank(group) ==34)
    %       break;
    %     end;
            ind_of_group = (clu_for_day==group_for_shank(group)).*valid_vecs{shank};
            spk_of_group = spk(:,:,ind_of_group~=0);
            if(size(spk_of_group,3)~=0)
                m = mean(spk_of_group,3);
                u = fft_upsample(m', 8 )';
                ind = min(get_index(u));

                idx1 = strcmp( sDavid.filename, folder_name );
                idx2 = ismember( sDavid.shankclu( :, [ 1 2 ] ), [shank group_for_shank(group)], 'rows' );
                idx = idx1 & idx2;
                disp([find(idx~=0) shank group_for_shank(group)]);
                sDavid.mean{find(idx~=0)} = u(ind,:);
            end
        end
    end
clearvars delim spk_path file_prefix clu_path C spk clu path i


