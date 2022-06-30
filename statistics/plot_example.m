function [eSTG1, eSTG2, act, sil, dcCCH, crCCH, cchbins, pred] = plot_example(path_a, path_b)
figure;
a = readNPY(path_a);
b = readNPY(path_b);
T = cat(1, a, b) * 20;
L = cat(1, ones(size(a)), ones(size(b)) * 2);
[eSTG1, eSTG2, act, sil, dcCCH, crCCH, cchbins, pred] = call_cch_stg( T, L, 20000, 0.001, 0.05, 11);
bar(cchbins,dcCCH(:,1) - pred(:,1), 1,'k')
return