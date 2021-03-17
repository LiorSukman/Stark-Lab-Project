function [ w ] = unif_ach_idx( cur_hist )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    idx = abs(cur_hist(:,1))<=30;
    upsample_hist = (fft_upsample(cur_hist(idx,2), 8 )');
    upsample_hist = upsample_hist.*(upsample_hist>=0);
    upsample_times = cur_hist(idx,1);
    upsample_times = linspace(upsample_times(1),upsample_times(end),length(upsample_hist));
    idx = upsample_times>=0;
    cdf_hist = cumsum(upsample_hist(idx));
    w = sum( (cdf_hist/max(cdf_hist)-linspace(0,1,length(cdf_hist)) ) ); % RMS distance of the CDF from the uniform dist      
%     disp(length(upsample_hist));
    w = w/length(upsample_hist);
end

