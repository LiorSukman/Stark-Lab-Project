function [ w ] = bump_idx( h_count )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    midband_peak = @(x) 10*log(40000*sum((x(floor(length(x)/2) + [320:2400])).^2));

    w = 81;
    smooth_hist = (conv(h_count, hann(w)./sum(hann(w)),'same'));
    psd = (abs(fftshift(fft(smooth_hist))));
    psd = psd./sum(psd);
    
    w = 91;
    smooth_psd = (conv(psd, hann(w)./sum(hann(w)),'same'));
	w = midband_peak(smooth_psd);


end

