function [ w ] = ach_jmp_idx( cur_hist )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

    w = 151;
    smooth_hist = (conv(cur_hist(:,2)', hann(w)./sum(hann(w)),'same'));
    idx = cur_hist(:,1)>50 & cur_hist(:,1)<1300;
    smooth_hist = smooth_hist(idx);
    arr = smooth_hist-linspace(smooth_hist(1),smooth_hist(end),length(smooth_hist));

	w = 100*log(sqrt( sum(arr.^2)/8000));
end

