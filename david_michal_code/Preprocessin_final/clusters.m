% clusters and sDavid needed
% this creates the mean and ACH for each cluster of the unsupervised model.
%%
mean_w = sDavid.mean;
hist_w = sDavid.hist;
time = linspace(-0.8,0.8,256);
figure();

for i=1:length(unique(clusters))-1

    idx = (clusters == i);
    ind_list = ind(idx~=0);
    num_red_in_cluster = sum(tags(idx)==1);
    num_blue_in_cluster = sum(tags(idx)==2);
    if(num_blue_in_cluster>num_red_in_cluster)
        rgb=[0 0 255]./255;
    else
        rgb=[255 0 0]./255;
    end
        
    cl.n_blue{i} = num_blue_in_cluster;
    cl.n_red{i} = num_red_in_cluster;
    cl.n{i} = length(ind_list);
%     cl.
    cl.unif_ach_idx_mean{i} = mean(sDavid.unif_ach_idx(ind(idx~=0)));
    cl.ach_risetime_mean{i} = mean(sDavid.ach_risetime(ind(idx~=0)));
    cl.ach_jmp_idx_mean{i} = mean(sDavid.ach_jmp_idx(ind(idx~=0)));
    cl.pds_feature_1_mean{i} = mean(sDavid.pds_feature_1(ind(idx~=0)));
    cl.pds_feature_2_mean{i} = mean(sDavid.pds_feature_2(ind(idx~=0)));
    cl.max_speed_mean{i} = mean(sDavid.max_speed(ind(idx~=0)));
    cl.fwhm_mean{i} = mean(sDavid.fwhm(ind(idx~=0)));
    cl.peak2peak_mean{i} = mean(sDavid.peak2peak(ind(idx~=0)));
    cl.break_measure_mean{i} = mean(sDavid.break_measure(ind(idx~=0)));
    cl.rise_coeff_mean{i} = mean(sDavid.rise_coeff(ind(idx~=0)));
    cl.smile_or_cry_mean{i} = mean(sDavid.smile_or_cry(ind(idx~=0)));
    cl.get_acc_mean{i} = mean(sDavid.get_acc(ind(idx~=0)));
    
    cl.unif_ach_idx{i} = std(sDavid.unif_ach_idx(ind(idx~=0)));
    cl.ach_risetime{i} = std(sDavid.ach_risetime(ind(idx~=0)));
    cl.ach_jmp_idx{i} = std(sDavid.ach_jmp_idx(ind(idx~=0)));
    cl.pds_feature_1{i} = std(sDavid.pds_feature_1(ind(idx~=0)));
    cl.pds_feature_2{i} = std(sDavid.pds_feature_2(ind(idx~=0)));
    cl.max_speed{i} = std(sDavid.max_speed(ind(idx~=0)));
    cl.fwhm{i} = std(sDavid.fwhm(ind(idx~=0)));
    cl.peak2peak{i} = std(sDavid.peak2peak(ind(idx~=0)));
    cl.break_measure{i} = std(sDavid.break_measure(ind(idx~=0)));
    cl.rise_coeff{i} = std(sDavid.rise_coeff(ind(idx~=0)));
    cl.smile_or_cry{i} = std(sDavid.smile_or_cry(ind(idx~=0)));
    cl.get_acc{i} = std(sDavid.get_acc(ind(idx~=0)));

        
    subplot(3,4,2*(i-1)+1);
    plot(time,mean(cell2mat(mean_w(ind(idx~=0))),1)*(4000/(2^15)),'color',rgb, 'LineWidth',2)
    title(['mean for cluster #',num2str(i)]);
    ylabel('volatge [\muV]');
    xlabel('time [ms]');
    axis tight;
    
    
    zero_hist  = hist_w{1};
    mean_hist = 0*cur_hist(:,2);
    for j=1:length(ind_list)
        cur_hist = hist_w{ind_list(j)};
        mean_hist = mean_hist + cur_hist(:,2)./length(ind_list);
    end
    
    
    subplot(3,4,2*(i-1)+2);
    idx = abs(zero_hist(:,1))<=50;
    bar(zero_hist(idx,1),mean_hist(idx), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
    title(['ACH for cluster #',num2str(i)]);
    xlabel('\tau [ms]');
    ylabel('counts');
    axis tight;
end
%%



