function [ exc, inh ] = calc_cc_res()
pairs = load('F:\Users\Lior\Desktop\University\Masters Degree\Stark Lab\Code\Stark Lab Project\cch_pairs.mat').pairs;
pairs_shape = size(pairs);
pairs_num = pairs_shape(1);
exc = zeros(pairs_num, 2);
inh = zeros(pairs_num, 2);
for i=1:pairs_num
    if mod(i, 1000) == 0
        disp(i);
    end
    a = readNPY(reshape(pairs(i, 1, :), 1, []));
    b = readNPY(reshape(pairs(i, 2, :), 1, []));
    T = cat(1, a, b) * 20;
    L = cat(1, ones(size(a)), ones(size(b)) * 2);
    % [eSTG1, eSTG2, act, sil, dcCCH, crCCH, cchbins] = call_cch_stg( T, L, 20000, 0.001, 0.05, 11);
    [~, ~, act, sil, ~, ~, ~] = call_cch_stg( T, L, 20000, 0.001, 0.05, 11);
    exc(i, :) = act;
    inh(i, :) = sil;
end
save('F:\Users\Lior\Desktop\University\Masters Degree\Stark Lab\Code\Stark Lab Project\cch_res.mat','exc','inh');
return
