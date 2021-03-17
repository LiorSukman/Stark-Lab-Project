function [ acc_param ] = get_acc( spike )
%get the second derivative parameter
    spike = spike./sum(spike.^2);
    roi = (128:170);
    w = (conv(spike, hann(15)./sum(hann(15)),'same'));
    w = diff(diff(w));
    w = w(10:end-10);
    w = (conv(w, hann(15)./sum(hann(15)),'same'));
    try
        acc_param = 20*log(sum((10^7*w(roi)).^2));
    catch
        acc_param = nan;
    end 
end

