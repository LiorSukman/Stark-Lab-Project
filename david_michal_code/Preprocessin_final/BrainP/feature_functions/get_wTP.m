function wTP = get_wTP(w)
    minVal = min(w);
    minInd = max(find(w == minVal)); %% we added max
    wNew = w(minInd:size(w,1));
    maxVal = max(wNew);
    maxInd = min(find(w == maxVal));%% we added max
    wTP = (maxInd - minInd)/(8*20);
end
    