function [stm_total_matrix] = marge_2_stm(path, stm_total)
% load data
    % path1 = 'C:\Users\mich\Google Drive\Brain project\data\es25nov11_3\es25nov11_3.stm.033';
    path1 = path;
    stm1 = load(path1, '-mat');
    times1 = stm1.stim.times;
    size1 = size(times1,1);
    i=1;
    % path2 = 'C:\Users\mich\Google Drive\Brain project\data\es25nov11_3\es25nov11_3.stm.034';
    % stm2 = load(path2, '-mat');
    % times2 = stm2.stim.times;
    times2 = stm_total;
    j = 1;
    size2 = size(times2,1);
    stm_total = {};

% run calc
    while (i <= size1) && (j <= size2)
        time1 = times1(i,:);
        start_time1 = time1(1);
        end_time1 = time1(2);

        time2 = times2(j,:);
        start_time2 = time2(1);
        end_time2 = time2(2);

        if start_time1 < start_time2
            if end_time1 < start_time2 % the time1 is separate from time2
                time = [start_time1,end_time1];
                i = i+1; % moving to next element in times1
                stm_total{end+1} = time;
            else 
                time = [start_time1,end_time2];
                times2(j,:) = time; %update the time range in times2
            end         
        elseif start_time2 < start_time1
                if end_time2 < start_time1 % the time2 is separate from time1
                    time = [start_time2,end_time2];
                    j = j+1; % moving to next element in times2
                    stm_total{end+1} = time;
                else 
                    time = [start_time2,end_time1];
                    times1(i,:) = time; %update the time range in times2
                end
        else
            if end_time1>end_time2
                time = [start_time1,end_time1];
            else
                time = [start_time1,end_time2];
            end
            stm_total{end+1} = time;
            j = j+1; % moving to next element in times2
            i = i+1; % moving to next element in times1
        end
    end 

    if i<size1
       for k=i:size1
            time1 = times1(k,:);
            start_time1 = time1(1);
            end_time1 = time1(2);
            stm_total{end+1} = [start_time1,end_time1]; 
       end
    end
    if j<size2
       for t=j:size2
            time2 = times2(t,:);
            start_time2 = time2(1);
            end_time2 = time2(2);
            stm_total{end+1} = [start_time2,end_time2];   
       end
    end

% convert from cell to martix 
    a = cell2mat(stm_total);
    re_size = size(a,2)/2;
    stm_total_matrix = reshape(a,[2,re_size]);
    stm_total_matrix = stm_total_matrix';
end 