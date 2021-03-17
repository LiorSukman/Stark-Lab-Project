function  diff_spike  = get_pos_half_diff( spike )
minVal = min(spike);
minInd = min(find(spike == minVal)); %% we added max
spikeNew = spike(minInd+1:size(spike,2));
% plot(spikeNew);
diff_spike = discrite_derivate(spikeNew);

% [spikeNew(2:end),0] - spikeNew;
% diff_spike = diff_spike(1:end-1);

end

