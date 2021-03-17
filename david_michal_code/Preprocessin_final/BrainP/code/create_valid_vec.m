function [valid_vec] = create_valid_vec(stm_total, res_i_path) 
    res = load(res_i_path);
    vec_size = size(res,1);
    valid_vec = ones(vec_size,1);
    stm_p = 1;
    stm_size = size(stm_total,1);
    i=1;
    while (i<vec_size) && (stm_p < stm_size)
        cur_time = res(i);
        time = stm_total(stm_p,:);
        start_time = time(1);
        end_time = time(2);
        
        if cur_time < start_time
            i = i+1;
            continue
        elseif start_time <=cur_time  && cur_time<=end_time
                valid_vec(i) = 0;
                i = i+1;
        else
            stm_p = stm_p+1;
        end
    end
    
end
