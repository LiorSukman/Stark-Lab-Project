function [fwhm] = get_fwhm(w)

    minVal = min(w);
    minInd = find(w == minVal, 1, 'first' ); %% we added max
    fwhm_val = minVal + 0.5*peak2peak(w);


    % Find where the data first drops below half the max.
    index1 = find(w <= fwhm_val, 1, 'first');
    % Find where the data last rises above half the max.
    index2 = find(w <= fwhm_val, 1, 'last');
    fwhm = (index2-index1 + 1)./(8*20); % FWHM in ms.
end
    