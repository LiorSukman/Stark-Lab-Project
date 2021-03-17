%% 
global sDavid;
num_of_days = length(sDavid.filename);
clu = sDavid.shankclu(:,2);
shank = sDavid.shankclu(:,1);
filenames = sDavid.filename;
for day=1:num_of_days
disp([filenames(day) shank(day) clu(day) ]);
end

%%
abs_path = pwd;
path = strcat(pwd,'\data\');
folder_name =   'es25nov11_3';
valid_vecs =  get_valid_vecs(strcat(path,folder_name));
cd(abs_path);

idx1 = strcmp( sDavid.filename, folder_name );
num_shanks = max(sDavid.shankclu(idx1,1));
for shank=1:num_shanks
%     disp(shank);
    clu_path = char(strcat(path,folder_name,'\',folder_name,'.clu.',num2str(shank)));
    res_path = char(strcat(path,folder_name,'\',folder_name,'.res.',num2str(shank)));
    clu = importdata(clu_path);
    res = importdata(res_path);
    clu(1) = [];
    
    
    idx2 = ismember( sDavid.shankclu( :, 1), shank );
    idx = idx1 .* idx2;
    group_for_shank = sDavid.shankclu(idx~=0,2);
%     disp(group_for_shank);
    for group=1:length(group_for_shank)
        
        
        idx1 = strcmp( sDavid.filename, folder_name );
        idx2 = ismember( sDavid.shankclu( :, [ 1 2 ] ), [shank group_for_shank(group)], 'rows' );
        idx = idx1 & idx2;
        disp([num2str(find(idx~=0)),'  ',folder_name,'  ',num2str(shank),' ', num2str(group_for_shank(group))]);
        clu_idx = ( clu == group_for_shank(group) ) & valid_vecs{shank};
        hist=my_cch(res(clu_idx));
%         figure();
%         bar(hist(:,1),hist(:,2));axis tight;
        sDavid.hist{find(idx~=0)} = hist;
    end
end


save('sDavid','sDavid');