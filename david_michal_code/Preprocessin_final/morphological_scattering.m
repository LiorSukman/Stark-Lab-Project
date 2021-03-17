%%
clear;
close all;
%%
sDavid = load('sDavid');
sDavid = sDavid.sDavid;
exc = sDavid.exc;
act = sDavid.act;
inh = sDavid.inh;
blue = act | inh;
red = exc==1;
untagged = ~red &~blue;

%% save features
num_of_days = length(sDavid.filename);
clu = sDavid.shankclu(:,2);
shank = sDavid.shankclu(:,1);
filenames = sDavid.filename;
for day=1:num_of_days
%     disp([filenames(day) shank(day) clu(day) ]);
    idx1 = strcmp( sDavid.filename, filenames(day) ); 
    idx2 = ismember( sDavid.shankclu( :, [ 1 2 ] ), [ shank(day) clu(day) ], 'rows' ); 
    idx = idx1 & idx2;
    w = sDavid.mean{day};
    if(~isempty(w))
        sDavid.fwhm(day) = get_fwhm(sDavid.mean{day}');
        sDavid.trouf2peak(day) = get_wTP(sDavid.mean{day}');
        sDavid.peak2peak(day) = peak2peak(sDavid.mean{day}');
        sDavid.rise_coeff(day) = rise_coeff(sDavid.mean{day});
        sDavid.get_acc(day) = get_acc(sDavid.mean{day}');
        sDavid.break_measure(day) =  break_measure(sDavid.mean{day}');
        sDavid.max_speed(day) = max_speed(sDavid.mean{day}');
        sDavid.smile_or_cry(day) = smile_or_cry(sDavid.mean{day}');
    else
        sDavid.fwhm(day) = NaN;
        sDavid.trouf2peak(day) = NaN;
        sDavid.peak2peak(day) = NaN;
        sDavid.rise_coeff(day) = NaN;
        sDavid.get_acc(day) = NaN;
        sDavid.break_measure(day) =  NaN;
        sDavid.max_speed(day) = NaN;
        sDavid.smile_or_cry(day) = NaN;
    end
end




%%
features = [sDavid.max_speed(~isnan(sDavid.max_speed))',...
            sDavid.fwhm(~isnan(sDavid.max_speed)),...
            sDavid.peak2peak(~isnan(sDavid.max_speed))' ,...
            sDavid.break_measure(~isnan(sDavid.max_speed))' ,...
            sDavid.smile_or_cry(~isnan(sDavid.max_speed))' ,...
            sDavid.rise_coeff(~isnan(sDavid.max_speed))' ,...
            sDavid.get_acc(~isnan(sDavid.max_speed))' ];
feature_names = 'max_speed fwhm peak2peak break-measure smile-or-cry  rise-coeff get_acc';
feature_names = strsplit(feature_names);
act = sDavid.act(~isnan(sDavid.max_speed));
inh = sDavid.inh(~isnan(sDavid.max_speed));
exc = sDavid.exc(~isnan(sDavid.max_speed));
%%
% save('features','features');
% save('feature_names','feature_names');
% save('act','act');
% save('inh','inh');
% save('exc','exc');
% save('red','red');
% save('blue','blue');


%% all combinations
num_of_features = 7;

figure();
pairs = nchoosek((1:num_of_features),2);
plot_ind = num_of_features*(pairs(:,1)-1)+pairs(:,2);
for pair=1:size(pairs,1)
    subplot(7,7,plot_ind(pair));
    disp([pairs(pair,:)]);
    hold on;
    scatter( features(red,pairs(pair,1)), features(red,pairs(pair,2)),[],'r','.');
    scatter( features(blue,pairs(pair,1)), features(blue,pairs(pair,2)),[],'b','.');

%     scatter((features(:,pairs(pair,1))+normrnd(0,3,1,sum(ind2))),(sDavid.fwhm(ind2)'+normrnd(0,3,1,sum(ind2))),[],'r','filled');
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
for feature_num = 1:7; 

    subplot(2,4,feature_num);
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
hold on;
% scatter((sDavid.trouf2peak(ind3)+normrnd(0,0.001,1,sum(ind3))),(sDavid.rise_coeff(ind3)+normrnd(0,0.001,1,sum(ind3))),[],'g','filled');
% scatter((sDavid.trouf2peak(ind1)+normrnd(0,0.001,1,sum(ind1))),(sDavid.rise_coeff(ind1)+normrnd(0,0.001,1,sum(ind1))),[],'b','filled');
% scatter((sDavid.trouf2peak(ind2)+normrnd(0,0.001,1,sum(ind2))),(sDavid.rise_coeff(ind2)+normrnd(0,0.001,1,sum(ind2))),[],'r','filled');
scatter((sDavid.trouf2peak+normrnd(0,0.001,1,1063)),(sDavid.rise_coeff+normrnd(0,0.001,1,1063)),[],'filled');

hold off;
title('scattring that shows scattering');
xlabel('trouph2peak');
ylabel('rise coeff');
% scatter((trouf2peak(ind3)'+normrnd(0,3,1,sum(ind3)))',(fwhm_array(ind3)'+normrnd(0,3,1,sum(ind3)))',[],'g','filled');
axis tight;
grid on;
hold off;

