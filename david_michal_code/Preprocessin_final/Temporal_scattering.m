%%
clear;
close all;

sDavid = load('sDavid');
sDavid = sDavid.sDavid;
exc = sDavid.exc;
act = sDavid.act;
inh = sDavid.inh;
blue = act | inh;
red = exc==1;
untagged = ~red &~blue;
%%
num_of_days = length(sDavid.filename);
clu = sDavid.shankclu(:,2);
shank = sDavid.shankclu(:,1);
filenames = sDavid.filename;
for day=1:length(sDavid.hist)
%     disp([filenames(day) shank(day) clu(day) ]);
    idx1 = strcmp( sDavid.filename, filenames(day) ); 
    idx2 = ismember( sDavid.shankclu( :, [ 1 2 ] ), [ shank(day) clu(day) ], 'rows' ); 
    idx = idx1 & idx2;
    cur_hist = sDavid.hist{day};
    if(~isempty(cur_hist) & sum(cur_hist(:,2))>=8000)
        sDavid.unif_ach_idx(day) = unif_ach_idx(sDavid.hist{day});
        sDavid.ach_risetime(day) = ach_risetime(sDavid.hist{day});
        sDavid.ach_jmp_idx(day) = ach_jmp_idx(sDavid.hist{day});
        sDavid.pds_feature_1(day) = pds_feature_1(sDavid.hist{day});
        sDavid.pds_feature_2(day) = pds_feature_2(sDavid.hist{day});

    else
        sDavid.unif_ach_idx(day) = NaN;
        sDavid.ach_risetime(day) = NaN;
        sDavid.ach_jmp_idx(day) = NaN;
        sDavid.pds_feature_1(day) = NaN;
        sDavid.pds_feature_2(day) = NaN;

    end
end

%% feature matrix
features = [sDavid.unif_ach_idx(~isnan(sDavid.unif_ach_idx))',...
            sDavid.ach_risetime(~isnan(sDavid.unif_ach_idx))',... 
            sDavid.ach_jmp_idx(~isnan(sDavid.unif_ach_idx))',...
            sDavid.pds_feature_1(~isnan(sDavid.unif_ach_idx))' ,...
            sDavid.pds_feature_2(~isnan(sDavid.unif_ach_idx))' ];
feature_names = 'unif-ach-idx ach-risetime ach-jmp-idx pds-feature-1 pds-feature-2'; 
feature_names = strsplit(feature_names);
act = sDavid.act(~isnan(sDavid.unif_ach_idx));
inh = sDavid.inh(~isnan(sDavid.unif_ach_idx));
exc = sDavid.exc(~isnan(sDavid.unif_ach_idx));
blue = act | inh;
red = exc==1;
untagged = ~red &~blue;

%% all combinations
num_of_features = 5;
figure(3);
pairs = nchoosek((1:num_of_features),2);
plot_ind = num_of_features*(pairs(:,1)-1)+pairs(:,2);
% pairs = pairs(5:7,:);
% pairs = vertcat([2 1],pairs);
for pair=1:size(pairs,1)
    subplot(4,5,plot_ind(pair));
    disp([pairs(pair,:)]);
    hold on;
    scatter( features(red,pairs(pair,1)), features(red,pairs(pair,2)),[],'r','.');
    scatter( features(blue,pairs(pair,1)), features(blue,pairs(pair,2)),[],'b','.');

    xlabel(['f#',num2str(pairs(pair,1))]);
    ylabel(['f#',num2str(pairs(pair,2))]);
    axis tight;
    
    grid on;
    hold off;
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);

end




%% PDFs
figure();
for feature_num = 1:5; 
%     
    subplot(1,5,feature_num);
     [red_pdf,red_edges] = histcounts(features(red,feature_num),100,'normalization','pdf');
    [blue_pdf,blue_edges] = histcounts(features(blue,feature_num),100,'normalization','pdf');
    % plot(x,p/sum(p)); %PDF
%     subplot(1,2,2);
    hold on;
    rgb = [0, 102, 255]./255;
%     bar(blue_pdf, 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
% plot(,N)
    plot(0.5*(blue_edges(1:end-1)+blue_edges(2:end)),blue_pdf,'b','LineWidth',2);
    rgb = [255, 153, 102]./255;
    plot(0.5*(red_edges(1:end-1)+red_edges(2:end)),red_pdf,'r','LineWidth',2);
%     bar(red_pdf, 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
    alpha(.75);
    title(['PDF of ',feature_names(feature_num)],'Interpreter', 'none');
%     legend('blue','red');
%     histogram(features(blue,feature_num),50,'normalization','pdf','BarWidth',1);
    hold off;
end


%% REDANDENCY
figure();
% ach_risetime 
% subplot(1,2,2);
hold on;
% scatter((sDavid.trouf2peak(ind3)+normrnd(0,0.001,1,sum(ind3))),(sDavid.rise_coeff(ind3)+normrnd(0,0.001,1,sum(ind3))),[],'g','filled');
% scatter((sDavid.trouf2peak(ind1)+normrnd(0,0.001,1,sum(ind1))),(sDavid.rise_coeff(ind1)+normrnd(0,0.001,1,sum(ind1))),[],'b','filled');
% scatter((sDavid.trouf2peak(ind2)+normrnd(0,0.001,1,sum(ind2))),(sDavid.rise_coeff(ind2)+normrnd(0,0.001,1,sum(ind2))),[],'r','filled');
scatter(features(red,1)+normrnd(0,0.001,1,sum(red))',features(red,2)+normrnd(0,0.001,1,sum(red))',[],'r','filled');
scatter(features(blue,1)+normrnd(0,0.001,1,sum(blue))',features(blue,2)+normrnd(0,0.001,1,sum(blue))',[],'b','filled');

title('tagged scattering');
xlabel('unif-ach-idx');
ylabel('ach-risetime');
% scatter((trouf2peak(ind3)'+normrnd(0,3,1,sum(ind3)))',(fwhm_array(ind3)'+normrnd(0,3,1,sum(ind3)))',[],'g','filled');
axis tight;
grid on;
hold off;

% % subplot(1,2,1);
% scatter(features(:,1)+normrnd(0,0.001,1,size(features(:,1),1))',features(:,2)+normrnd(0,0.001,1,size(features(:,2),1))',[],'filled');
% scatter(features(:,1)+normrnd(0,0.001,1,size(features(:,1),1))',features(blue,2)+normrnd(0,0.001,1,size(features(:,1),1))',[],'b','filled');

hold off;
title('scattring that shows scattering');
xlabel('unif-ach-idx');
ylabel('ach-risetime');
% scatter((trouf2peak(ind3)'+normrnd(0,3,1,sum(ind3)))',(fwhm_array(ind3)'+normrnd(0,3,1,sum(ind3)))',[],'g','filled');
axis tight;
grid on;
hold off;

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


% %% all combinations
% num_of_features = 5;
% 
% figure(3);
% pairs = nchoosek((1:num_of_features),2);
% plot_ind = num_of_features*(pairs(:,1)-1)+pairs(:,2);
% for pair=1:size(pairs,1)
%     subplot(4,5,plot_ind(pair));
%     disp([pairs(pair,:)]);
%     hold on;
%     scatter( features(red,pairs(pair,1)), features(red,pairs(pair,2)),[],'r','.');
%     scatter( features(blue,pairs(pair,1)), features(blue,pairs(pair,2)),[],'b','.');
% 
% %     scatter((features(:,pairs(pair,1))+normrnd(0,3,1,sum(ind2))),(sDavid.fwhm(ind2)'+normrnd(0,3,1,sum(ind2))),[],'r','filled');
%     xlabel(feature_names(pairs(pair,1)));
%     ylabel(feature_names(pairs(pair,2)));
%     axis tight;
%     
%     grid on;
%     hold off;
% 
% end
% %%
% 
% %%
% 
% %%
% for feature_num = 1:13; 
% %     figure();
%     subplot(3,5,feature_num);
%     red_pdf = histcounts(features(red,feature_num),50,'normalization','pdf');
%     blue_pdf = histcounts(features(blue,feature_num),50,'normalization','pdf');
%     % plot(x,p/sum(p)); %PDF
% %     subplot(1,2,2);
%     hold on;
%     rgb = [0, 102, 255]./255;
%     bar(blue_pdf, 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
%     rgb = [255, 153, 102]./255;
%     bar(red_pdf, 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
%     alpha(.75);
%     title(feature_names(feature_num));
% %     legend('blue','red');
% %     histogram(features(blue,feature_num),50,'normalization','pdf','BarWidth',1);
%     hold off;
% end
% 
