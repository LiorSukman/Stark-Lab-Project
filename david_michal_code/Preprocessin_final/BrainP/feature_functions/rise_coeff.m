function [ rise_coeff ] = rise_coeff( w )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% window_parameter = 40;
% diff_spike  = get_pos_half_diff( spike );
% try
%     rise_coeff = sum(diff_spike(1:window_parameter));
% catch
%     rise_coeff = nan;
% end 
w = w./sum(w.^2);
w = (conv(w, hann(5)./sum(hann(5)),'same'));
y = w(size(w,2)/2:end) -linspace(w(size(w,2)/2),w(end),length(size(w,2)/2));
maxVal=max(y);
rise_coeff = min(find(y == maxVal))/(20*8);
end

