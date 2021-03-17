function hist =my_cch(res)


    zrLag=0;
    jitterMs = 0.5; % bin size for correlation plot
    nLagsMs = 1500; % show 30 msec
    jitterSamp = round(20000/1000); % 1 ms
    nLags = round(nLagsMs/jitterMs);
    iTimes = int32(double(res)/jitterSamp);
    jTimes = int32(double(res)/jitterSamp);
    % count agreements of jTimes + lag with iTimes
    lagSamp = -nLags:nLags;
    intCount = zeros(size(lagSamp));
    for iLag = 1:numel(lagSamp)
        if (zrLag==0 && lagSamp(iLag)==0)
            continue;
        end
        intCount(iLag) = numel(intersect(iTimes, jTimes + lagSamp(iLag)));
    end

    lagTime = lagSamp*jitterMs;
rgb=[49 88 165]./255;
% bar(lagTime, intCount,'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1)
hist = [lagTime', intCount'];
end