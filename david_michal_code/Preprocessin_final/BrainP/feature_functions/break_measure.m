function [break_measure] = break_measure(w)

    % this measures whether there is a bump rigt before the spiking.
    % only part of the exc==1 spikes exibits this feature.
    w=w./sum(w.^2);    
    roi = (65:95);
    w = (conv(w, hann(15)./sum(hann(15)),'same'));
    w = diff(diff(w(20:end-10)));
    w = (conv(w, hann(5)./sum(hann(5)),'same'));

    break_measure = 100*log(10^12*sum(w.^2));
    

end
    