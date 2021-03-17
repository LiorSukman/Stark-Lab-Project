%% creat list of folders:
clear;
close all;
%%
dir_name = regexp(pwd,'\','split');
dir_name = dir_name(end);
if(strcmp(dir_name,'Preprocessin_final')==0)
      error('the path should be the drive "Brain project"');
end
abs_path = pwd; 
clearvars dir_name
%%
% fid = fopen('C:\Users\David Gertskin\Google Drive\Brain project\currDataList.txt');
fid = fopen(strcat(pwd,'\currDataList.txt'));
tline = fgetl(fid);
list_of_folders = {};
i = 0;

while ischar(tline)
    list_of_folders{end+1} = tline;
    tline = fgetl(fid);
end
fclose(fid);
clearvars tline i fid ans

%% LOAD DATA and get processed data
num_channels = 8;
num_samples = 32;
num_of_folders = length(list_of_folders);
processed_data = {};
%% UPDATE the sDavid data structure according to the new sessions needed to be processed
global sDavid;
if exist('sDavid.mat', 'file') == 2
     % File exists.
     sDavid = load('sDavid.mat');
     sDavid = sDavid.sDavid;
else
     % File does not exist.
     disp(' no sDavid - file created ');
     L = load( strcat(pwd,'\CelltypeClassification.mat'));
     sDavid = L.sPV;
     sDavid = rmfield( sDavid, { 'slevel', 't', 'd', 'pidx', 'inhibited', 'excited', 'hw', 'asy', 'acom', 'pinfo', 'pstats', 'pstats0', 'pidx0' } );
     sDavid.shankclu( :, 3 ) = [];
     sDavid.fwhm = NaN * ones( size( sDavid.inh ) );
     sDavid.mean = cell(size( sDavid.inh ) );

end

%% RUN the calculation for all the folders.
% the calculation repeats for session names that are already in the data structure
% logic can be improved.

for i=1:num_of_folders
   cd(abs_path)
   folder_name = [list_of_folders{i}];
   path = [strcat(pwd,'\data\') folder_name];
   disp(path);
   disp(path);
   get_data(path, folder_name);
end 
clearvars num_channels num_samples path folder_name i 
%%
cd(abs_path);

