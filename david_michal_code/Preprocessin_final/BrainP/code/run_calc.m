function [features, tags, data]= run_calc( path,folder_name )
num_channels = 8;
num_samples = 32;

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

sst=importdata(char(strcat(path,file_prefix,'.sst')));
num_shanks = max(sst.shankclu(:,1));
data.spk = {};
for i = 1:num_shanks
    spk_path = char(strcat(path,file_prefix,'.spk.',num2str(i)));
    try
        spk = readspk(spk_path,num_channels, num_samples,[] , 'int16', 'single');
    catch
        disp(file_prefix);
    end 
    data.spk{i} = spk;
    disp(char(strcat(path,file_prefix,'.spk.',num2str(i))));
    
    clu_path = char(strcat(path,file_prefix,'.clu.',num2str(i)));
    clu = importdata(clu_path);
    data.clu{i} = clu(2:end);
    disp(clu_path); 
    data.clu_ind{i} = find(sst.shankclu(:,1) == i);
    
end
clearvars delim spk_path file_prefix clu_path C spk clu path i

%%  GET VALID SPIKES

% a valid spike is a spike that belongs to a groups for which the Lration
% paramteter is less then 
Lratio_threshold = 0.5;
ISI_threshold = 0.02;
L = load( 'C:\Users\mich\Google Drive\Brain project\BrainP\CelltypeClassification (1).mat');

tag_index = find(strcmp(L.sPV.filename , folder_name));
tag_shankclu = L.sPV.shankclu(tag_index,1:2);
tag = [tag_index, tag_shankclu];

% a = [1 5 2 5 3 5 4 5]
% b = [2,3,4]
% % Give "true" if the element in "a" is a member of "b".
% c = ismember(a, b)
% % Extract the elements of a at those indexes.
% indexes = find(c)


for i = 1:num_shanks
    data.Lratio{i} = sst.Lratio(data.clu_ind{i});
    data.ISIindex{i} = sst.ISIindex(data.clu_ind{i});
    data.groups{i} = sst.shankclu(data.clu_ind{i},2);
    data.groups{i} = data.groups{i}.* ((data.Lratio{i} < Lratio_threshold) | (data.ISIindex{i} < ISI_threshold) );
    
    pre_index_for_shank_i = ismember(tag(:,2),i);
    tags_group = tag(pre_index_for_shank_i,1:3); 
    index_shank_group_with_tag_e_m = ismember(data.groups{i},tags_group(:,3));
    index_shank_group_with_tag_m_e = ismember(tags_group(:,3),data.groups{i});
    

    data.act{i} = index_shank_group_with_tag_e_m;
    data.inh{i} = index_shank_group_with_tag_e_m;
    data.exc{i} = index_shank_group_with_tag_e_m;
    x = tags_group(:,1:1);
    data.act{i}(index_shank_group_with_tag_e_m) = L.sPV.act(x(index_shank_group_with_tag_m_e));
    data.inh{i}(index_shank_group_with_tag_e_m) = L.sPV.inh(x(index_shank_group_with_tag_m_e));
    data.exc{i}(index_shank_group_with_tag_e_m) = L.sPV.exc(x(index_shank_group_with_tag_m_e));

    
% ADD TO THE NEW CALC
    data.pyr{i} = sst.pyr(data.clu_ind{i});
    groups = data.groups{i};
    data.groups{i} = groups(data.groups{i}>0); 
% groups_1 are the correctly seperated groups .
    data.number_of_groups{i} = size(groups,1);
    data.filterd_clu{i} = data.clu{i}.*ismember(data.clu{i},groups);
end

clearvars Lratio_threshold  Lratio_1 i groups
%% GET FILTERED SPK
for i = 1:num_shanks
    [valid_indices,~,~] = find(data.filterd_clu{i});
    spk = data.spk{i};
    data.filtered_spk{i} = spk(:,:,valid_indices);
    filterd_clu = data.filterd_clu{i};
    data.filterd_clu{i} = filterd_clu(filterd_clu ~= 0);
end

clearvars spk i filterd_clu valid_indices


%% GET REDUCED SPK

upsample_factor = 8;

for j = 1:num_shanks
    sample_size = size(data.filterd_clu{j},1);
    reduced_spk = zeros(num_samples*upsample_factor,sample_size);
    filtered_spk = data.filtered_spk{j};
    for i = 1:sample_size
        if(mod(i,10000)==0)
            fprintf('num of shank: %d, iter: %d\n',j,i);
        end
        ind = min(get_index(filtered_spk(:,:,i)));
        reduced_spk(:,i) = fft_upsample(filtered_spk(ind,:,i)', upsample_factor )'; %spk(ind,:,i);
        % normalization of the spike.
        factor = sqrt(sum(reduced_spk(:,i).^2)); % normalize by the norm
        reduced_spk(:,i) = reduced_spk(:,i)./factor;
    end  
    data.reduced_spk{j} = reduced_spk;
end

clearvars  i j ind factor filtered_spk sample_size


%% GET MEAN WAVEFORM

% avarage the representative spike (from the reduced spk file)

for j = 1:num_shanks
 sample_size = size(data.filterd_clu{j},1);
%     clu = data.clu{j};
    reduced_spk = data.reduced_spk{j};
    mean = zeros(data.number_of_groups{j}+1,size(reduced_spk,1));
    filterd_clu = data.filterd_clu{j};
    for i = 1:data.number_of_groups{j}+1
       [valid_indices,~,~] = find(filterd_clu==i);
       if ~isempty(valid_indices)
        number_of_elements = size(reduced_spk(:,valid_indices),2);
        mean(i,:) = mean(i,:) + (1.0/number_of_elements).*sum(reduced_spk(:,valid_indices),2)';
       end
       
    end
    data.mean{j} = mean(data.groups{j},:);
end


clearvars number_of_elements valid_indices i j mean filterd_clu reduced_spk sample_size


%% GET FEATURES


num_of_valid_groups = 0;
for j = 1:num_shanks
    num_of_valid_groups = num_of_valid_groups+ size(data.groups{j},1);
end

peak2peak_array = zeros(num_of_valid_groups,1);
trouf2peak = zeros(num_of_valid_groups,1);
fwhm_array = zeros(num_of_valid_groups,1);
rise_coeff_array = zeros(num_of_valid_groups,1);
sec_diff_array = zeros(num_of_valid_groups,1);
pyr = [];
s = 1;
for j = 1:num_shanks
    n_pyr = data.pyr{j};
    pyr = [pyr,(n_pyr(data.groups{j}-1)' )];
    for i = 1:size(data.groups{j},1)
        mean = data.mean{j};
        peak2peak_array(s) = peak2peak(mean(i,:));
        trouf2peak(s) = get_wTP(mean(i,:)');
        fwhm_array(s) = get_fwhm(mean(i,:)');
        rise_coeff_array(s) = rise_coeff(mean(i,:));
        sec_diff_array(s) = get_acc(mean(i,:));
        s = s + 1;
    end
   
end

%features = {peak2peak_array, trouf2peak, fwhm_array, rise_coeff_array,sec_diff_array};
features.peak2peak_array = peak2peak_array;
features.trouf2peak = trouf2peak;
features.fwhm_array = fwhm_array;
features.rise_coeff_array = rise_coeff_array;
features.sec_diff_array = sec_diff_array;
features.pyr = pyr;
features = [ peak2peak_array, fwhm_array
clearvars n_pyr s i j num_of_valid_groups mean

%% create tag_file

act = [];
inh = [];
exc = [];
for j = 1:num_shanks
    n_act = data.act{j};
    act = [act,(n_act(data.groups{j}-1)' )];
    n_inh = data.inh{j};
    inh = [inh,(n_inh(data.groups{j}-1)')];
    n_exc = data.inh{j};
    exc = [exc,(n_exc(data.groups{j}-1)')];
end
tags.act = act;
tags.inh = inh;
tags.exc = exc;


