function [valid_vecs] =  get_valid_vecs(path)
%%
%     cd
    % path = 'C:\Users\mich\Google Drive\Brain project\data\es25nov11_3';
    cd(path);
    listing = dir('*stm.0*');
    list_stm_names  = {listing.name};
    num_of_list_str_names =  size(list_stm_names,2);
    stm_total = {}; 
%     cd('C:\Users\mich\Google Drive\Brain project\BrainP');

%% 
    for i=1:num_of_list_str_names
        stm_i_path =[path '\' cell2mat(list_stm_names(i))];
        [stm_total] = marge_2_stm(stm_i_path, stm_total);
    end 
    clearvars stm_i_path
%% create validation vacrot
    cd(path);
    listing = dir('*res.*');
    list_res_names = {listing.name};
    num_of_list_res_names =  size(list_res_names,2);

%     cd('C:\Users\mich\Google Drive\Brain project\BrainP');
    valid_vecs = {};
    for i=1:num_of_list_res_names
        res_i_path =[path '\' cell2mat(list_res_names(i))];
        [valid_vec] = create_valid_vec(stm_total, res_i_path);
        valid_vecs{end+1} = valid_vec;
    end 

    clearvars res_i_path valid_vec listing list_res_names num_of_list_res_names

end