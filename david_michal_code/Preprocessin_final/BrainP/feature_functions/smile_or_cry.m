function [smile_or_cry] = smile_or_cry(spike)
    w = spike./sum(spike.^2);
    w = (conv(w, hann(15)./sum(hann(15)),'same'));
    w = (diff(w));
    w = w(10:end-10);
    roi = (170:230);
    smile_or_cry = 10^6*sum(w(roi));
end
    