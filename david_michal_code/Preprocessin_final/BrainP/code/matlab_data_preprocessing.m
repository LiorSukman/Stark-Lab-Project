%% dependencies
% get_fwhm.m
% get_index.m
% lin_expand.m
% readspk.m
% get_wTP.m
% fft_upsample.m
% 
%% LOAD DATA
clear;
close all;
num_channels = 8;
num_samples = 32;
path = 'C:\Users\mich\Google Drive\אלקטרוניקה1\שטויות של מיכל\BrainP\es25nov11_3-20190314T094526Z-001\es25nov11_3';

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

sst=importdata(char(strcat(path,file_prefix,'.sst')));
num_shanks = max(sst.shankclu(:,1));
data.spk = {};
for i = 1:num_shanks
    spk_path = char(strcat(path,file_prefix,'.spk.',num2str(i)));
    spk = readspk(spk_path,num_channels, num_samples,[] , 'int16', 'single');
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
for i = 1:num_shanks
    data.Lratio{i} = sst.Lratio(data.clu_ind{i});
    data.ISIindex{i} = sst.ISIindex(data.clu_ind{i});
    data.groups{i} = sst.shankclu(data.clu_ind{i},2);
    data.groups{i} = data.groups{i}.* ((data.Lratio{i} < Lratio_threshold) | (data.ISIindex{i} < ISI_threshold) );
    data.pyr{i} = sst.pyr(data.clu_ind{i});
    groups = data.groups{i};
    data.groups{i} = groups(data.groups{i}>0); 
% groups_1 are the correctly seperated groups.
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
%%  DRAW RANDOM SIGNALS 
filtered_spk = data.filtered_spk{2};
random_spike_ind = randperm(size(filtered_spk,3),9);
figure();
x = ones(1,10);
for n = 1:9
    subplot(3,3,n);
    plot(filtered_spk(:,:,random_spike_ind(n))');
    title(['random spike #',num2str(n)]);
end
clearvars random_spike_ind n filtered_spk x;


%% GET REDUCED SPK

upsample_factor = 8;

for j = 1:num_shanks
    sample_size = size(data.filterd_clu{j},1);
    reduced_spk = zeros(num_samples*upsample_factor,sample_size);
    filtered_spk = data.filtered_spk{j};
    for i = 1:sample_size
        if(mod(i,1000)==0)
            fprintf('iteration %d\n',i);
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

%%  DRAW RANDOM SIGNALS 
reduced_spk = data.reduced_spk{2};
random_spike_ind = randperm(size(reduced_spk,2),9);
figure();

for n = 1:9
    subplot(3,3,n)
    plot(reduced_spk(:,random_spike_ind(n))');
    title(['processed spike #',num2str(n)]);
end
clearvars random_spike_ind n reduced_spk;

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
%% PRINT THE MEAN SPIKE FOR EVERY SHANK
for j = 1:num_shanks
    mean = data.mean{j};
    figure();

    for n = 1:size(mean,1)
        subplot(6,6,n)
        plot(mean(n,:));
    end
end
clearvars sample_size clu reduced_spk j n mean
%% GET FEATURES


num_of_valid_groups = 0;
for j = 1:num_shanks
    num_of_valid_groups = num_of_valid_groups+ size(data.groups{j},1);
end

peak2peak_array = zeros(num_of_valid_groups,1);
trouf2peak = zeros(num_of_valid_groups,1);
fwhm_array = zeros(num_of_valid_groups,1);
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
        s = s + 1;
    end
   
end

clearvars n_pyr s i j num_of_valid_groups mean
%% SCATTER PLOT
figure();
subplot(2,1,1)
c = (255.0/36)*pyr;
scatter(trouf2peak',fwhm_array',[],pyr,'filled');
xlabel('trouf2peak');
ylabel('fwhm');
title('scatter plot fwhm vs. t2p');
subplot(2,1,2)
scatter(peak2peak_array',fwhm_array',[],c,'filled');
xlabel('peak2peak');
ylabel('fwhm');
title('scatter plot fwhm vs. p2p');
clearvars c



