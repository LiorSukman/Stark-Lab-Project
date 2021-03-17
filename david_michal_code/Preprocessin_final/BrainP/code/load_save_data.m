path  = 'C:\Users\mich\Google Drive\Brain project\BrainP\save_data';
cd(path);
listing = dir('es*.mat*');
list_names  = {listing.name};
num =  size(list_names,2);

%% Load data
for i=1:num
   path_for_day = [path '\' cell2mat(list_names(i))];
   load(path_for_day);
end

%% Create processed_data struct
% processed_data={};
% for i=1:num
%     name  = cell2mat(list_names(i));
%     processed_data = [processed_data name];
%  
% end
processed_data = {es04feb12_1 es09feb12_2 es09feb12_3 es17mar12_2 es20may12_1 es21may12_1 es25nov11_3 es25nov11_5 es25nov11_9};
cd('C:\Users\mich\Google Drive\Brain project\BrainP');
