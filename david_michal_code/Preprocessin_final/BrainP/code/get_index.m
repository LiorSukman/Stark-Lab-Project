function ind = get_index(w)
num_channels = size(w,1);
p2p = zeros(num_channels,1);
% mid = size(w,2)/2;
for i = 1:num_channels
    minVal = min(w(i,:));
    minInd = max(find(w(i,:) == minVal)); %% we added max
    wNew = w(i,minInd+1:size(w,2));
    maxVal = max(wNew);
%     maxInd = min(find(w(i,:) == maxVal));%% we added max
    p2p(i) = maxVal - minVal;
end
ind = find(p2p == max(p2p));
end
    