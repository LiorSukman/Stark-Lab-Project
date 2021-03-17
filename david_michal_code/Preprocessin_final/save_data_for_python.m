%%
% RUN FIRST BOTH:
%   * morphological_scattering.m
%   * temporal_scattering.m
%%
abs_path =pwd;
data_for_python_loc = strcat(abs_path,'\data for python');
cd(data_for_python_loc);
%%

%%
act = sDavid.act;
inh = sDavid.inh;
exc = sDavid.exc;
blue = act | inh;
red = exc==1;
ind = ~isnan(sDavid.unif_ach_idx) & ~(act & exc)';
untagged = ~red &~blue;
features = [sDavid.unif_ach_idx(ind)',...
            sDavid.ach_risetime(ind)',...
            sDavid.ach_jmp_idx(ind)',...
            sDavid.pds_feature_1(ind)' ,...
            sDavid.pds_feature_2(ind)',...
            sDavid.max_speed(ind)',...
            sDavid.fwhm(ind),...
            sDavid.peak2peak(ind)' ,...
            sDavid.break_measure(ind)' ,...
            sDavid.rise_coeff(ind)' ,...
            sDavid.smile_or_cry(ind)' ,...
            sDavid.get_acc(ind)' ];

feature_names = 'unif-ach-idx ach-risetime ach-jmp-idx pds-feature-1 pds-feature-2 max-speed fwhm peak2peak break-measure rise-coeff smile-or-cry get-acc'; 
feature_names = strsplit(feature_names);
act = act(ind);
inh = inh(ind);
exc = exc(ind);
blue = blue(ind);
red = red(ind);

%%
save('features','features');
save('feature_names','feature_names');
save('act','act');
save('inh','inh');
save('exc','exc');
save('red','red');
save('blue','blue');
cd(abs_path);