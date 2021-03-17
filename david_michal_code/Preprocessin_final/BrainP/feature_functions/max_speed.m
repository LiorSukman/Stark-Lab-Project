function [max_speed] = max_speed(w)
% this should be renamed. it quantifies how much time the spikes gain
% voltage after the peak.
%     w = w./sum(w);
    w = (conv(w, hann(15)./sum(hann(15)),'same'));
    w= diff(w);
    ind = find(w>w(131));
    ind = ind(ind>131);
     max_speed = length(ind)/(8*20);
end
    