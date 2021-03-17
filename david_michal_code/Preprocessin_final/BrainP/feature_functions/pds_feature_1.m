function chi = pds_feature_1( cur_hist )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    idx = abs(cur_hist(:,1))<=20000;
    w = 81;
    us_factor = 8;
    upsample_hist = (fft_upsample(cur_hist(idx,2), us_factor )');
    upsample_hist = upsample_hist.*(upsample_hist>=0);
    upsample_times = cur_hist(idx,1);
    upsample_times = linspace(upsample_times(1),upsample_times(end),length(upsample_hist));
    smooth_hist = (conv(upsample_hist, hann(w)./sum(hann(w)),'same'));
%     f_sample is 20kh
    f= linspace(-10000,10000,length(upsample_times));
    psd = (abs(fftshift(fft(smooth_hist))));
    w = 91;
    smooth_psd = (conv(psd, hann(w)./sum(hann(w)),'same'));
    chi = sum( f(f>=0).*smooth_psd(f>=0)./sum(smooth_psd(f>=0)));

end

