function [ w ] = ach_risetime( cur_hist )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    idx = abs(cur_hist(:,1))<=30;
    upsample_hist = (fft_upsample(cur_hist(idx,2), 8 )');
    upsample_hist = upsample_hist.*(upsample_hist>=0);
    upsample_times = cur_hist(idx,1);
    upsample_times = linspace(upsample_times(1),upsample_times(end),length(upsample_hist));
    idx = upsample_times>=0;
    upsample_times = upsample_times(idx);
    cdf_hist = cumsum(upsample_hist(idx));
    factor = (1/100);
    w = upsample_times(sum((cdf_hist<=factor.*max(cdf_hist))));


end

