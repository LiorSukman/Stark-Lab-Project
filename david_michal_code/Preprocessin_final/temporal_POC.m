%%
clear;
close all;
%%
sDavid = load('sDavid');
sDavid = sDavid.sDavid;
%%

%%
hist = sDavid.hist;
exc = sDavid.exc;
act = sDavid.act;
inh = sDavid.inh;
blue = act | inh;
red = exc==1;
untagged = ~red &~blue;

%% ACH

figure();
plot_num = 1;
for j = 1:400

    cur_hist = hist{j};
    if(isempty(cur_hist))
        continue;
    end
    if ~untagged(j) & sum(cur_hist(:,2))>=100000 & plot_num<=12
        subplot(3,4,plot_num);
        hold on;
        if(red(j))
            rgb=[255 0 0]./255;
            bar(cur_hist(:,1),cur_hist(:,2),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
            axis tight;   
        else
            rgb=[0 0 255]./255;
            bar(cur_hist(:,1),cur_hist(:,2), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
            axis tight;
        end
        xlabel('\tau [ms]');
        ylabel('counts');
        if(plot_num<=8)
            set(gca,'XTick',[]);
            xlabel([]);

        end
        if(~(plot_num == 1 | plot_num == 5 | plot_num == 9))

            set(gca,'YTick',[]);
            ylabel([]);
        end
        grid on;
        hold off;

        title([sDavid.filename{j},' ',num2str(sDavid.shankclu(j,1)),' ',...
                num2str(sDavid.shankclu(j,2))], 'Interpreter', 'none');

        plot_num = plot_num + 1;
    end
end
%% bad ACH curve
figure();
plot_num = 1;
for j = 1:800

    cur_hist = hist{j};
    if(isempty(cur_hist))
        continue;
    end
    if  sum(cur_hist(:,2))<=1000 & plot_num<=12
        subplot(3,4,plot_num);
        hold on;
        rgb=[49 88 165]./255;
        bar(cur_hist(:,1),cur_hist(:,2),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
       
        grid on;
        hold off;
%         title(['ACH example #',num2str(plot_num)]);
        title(['ACH for |res|= ',num2str(sum(cur_hist(:,2)))]);
        title([sDavid.filename{j},' ',num2str(sDavid.shankclu(j,1)),' ',...
                num2str(sDavid.shankclu(j,2)),'  |res|= ',num2str(sum(cur_hist(:,2))) ], 'Interpreter', 'none');
        xlabel('\tau [ms]');
        ylabel('counts');
        plot_num = plot_num + 1;
    end
end
    
%% close up ACH

figure();
plot_num = 1;
for j = 1:400

    cur_hist = hist{j};
    if(isempty(cur_hist))
        continue;
    end
    if ~untagged(j) & sum(cur_hist(:,2))>=100000 & plot_num<=12
        subplot(3,4,plot_num);
        hold on;
        idx = abs(cur_hist(:,1))<=30;
        if(red(j))
            rgb=[255 0 0]./255;
            bar(cur_hist(idx,1),cur_hist(idx,2),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
            axis tight;   
        else
            rgb=[0 0 255]./255;
            bar(cur_hist(idx,1),cur_hist(idx,2), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
            axis tight;
        end
        grid on;
        hold off;
%         title(['ACH example #',num2str(plot_num)]);
        xlabel('\tau [ms]');
        ylabel('counts');
        if(plot_num<=8)
            set(gca,'XTick',[]);
            xlabel([]);
% set(gca,'YTick',[])
        end
        if(~(plot_num == 1 | plot_num == 5 | plot_num == 9))
%             set(gca,'XTick',[])
            set(gca,'YTick',[]);
            ylabel([]);
        end
                title([sDavid.filename{j},' ',num2str(sDavid.shankclu(j,1)),' ',...
                num2str(sDavid.shankclu(j,2))], 'Interpreter', 'none');
        plot_num = plot_num + 1;
    end
end

%% BATCH OF 3 ACH

figure();
g = [24,25,26];

for j=1:3
    cur_hist = hist{g(j)};
    subplot(1,3,j);
%     bar((fft_upsample(h_count(roi), 4 )'));
    idx = abs(cur_hist(:,1))<=30;
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(cur_hist(idx,1),cur_hist(idx,2),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(cur_hist(idx,1),cur_hist(idx,2), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
        xlabel('\tau [ms]');
        ylabel('count');
    title([sDavid.filename{g(j)},' ',num2str(sDavid.shankclu(g(j),1)),' ',...
                num2str(sDavid.shankclu(g(j),2))], 'Interpreter', 'none');
    axis tight;
end

%% upsample ACH
figure();
g = [24,25,26];
for j=1:3
    cur_hist = hist{g(j)};
    
    idx = abs(cur_hist(:,1))<=30;
    upsample_hist = (fft_upsample(cur_hist(idx,2), 8 )');
    upsample_hist = upsample_hist.*(upsample_hist>=0);
    upsample_times = cur_hist(idx,1);
    upsample_times = linspace(upsample_times(1),upsample_times(end),length(upsample_hist));
    subplot(3,2,2*(j-1)+1);
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(cur_hist(idx,1),cur_hist(idx,2),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(cur_hist(idx,1),cur_hist(idx,2), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
    title(['original ACH']);
    subplot(3,2,2*(j-1)+2);
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(upsample_times,upsample_hist,'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(upsample_times,upsample_hist, 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
    title(['upsampled ACH']);
    axis tight

end

%% CDF ACH
figure();
g = [24,25,26];
for j=1:3
    cur_hist = hist{g(j)};
    idx = abs(cur_hist(:,1))<=30;
    upsample_hist = (fft_upsample(cur_hist(idx,2), 8 )');
    upsample_hist = upsample_hist.*(upsample_hist>=0);
    upsample_times = cur_hist(idx,1);
    upsample_times = linspace(upsample_times(1),upsample_times(end),length(upsample_hist));
    subplot(3,3,3*(j-1)+1);
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(cur_hist(idx,1),cur_hist(idx,2),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(cur_hist(idx,1),cur_hist(idx,2), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
%     title(['original ACH']);
    title([sDavid.filename{g(j)},' ',num2str(sDavid.shankclu(g(j),1)),' ',...
                num2str(sDavid.shankclu(g(j),2))], 'Interpreter', 'none');
    xlabel('\tau [ms]');
    ylabel('count');
    subplot(3,3,3*(j-1)+2);
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(upsample_times,upsample_hist,'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(upsample_times,upsample_hist, 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
    title(['upsampled ACH']);
    axis tight;
     xlabel('\tau [ms]');
    ylabel('count');   
    idx = upsample_times>=0;
    subplot(3,3,3*(j-1)+3);
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(upsample_times(idx),cumsum(upsample_hist(idx)),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(upsample_times(idx),cumsum(upsample_hist(idx)), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
    title(['CDF ACH']);
    axis tight;
        xlabel('\tau [ms]');
    ylabel('count');
end

%% UNIF DIST IDX
figure();
g = [24,25,26];
for j=1:3
    cur_hist = hist{g(j)};
    idx = abs(cur_hist(:,1))<=30;
    upsample_hist = (fft_upsample(cur_hist(idx,2), 8 )');
    upsample_hist = upsample_hist.*(upsample_hist>=0);
    upsample_times = cur_hist(idx,1);
    upsample_times = linspace(upsample_times(1),upsample_times(end),length(upsample_hist));
    idx = upsample_times>=0;
    cdf_hist = cumsum(upsample_hist(idx));
    subplot(1,3,j);
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(upsample_times(idx),cdf_hist/max(cdf_hist),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(upsample_times(idx),cdf_hist/max(cdf_hist), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
    hold on;
            rgb = [ 46 204 5 ]./255;
    plot(upsample_times(idx),linspace(0,1,length(cdf_hist)),'color',rgb, 'LineWidth',2);
    hold off;
    xlabel('\tau [ms]');
    ylabel('CDF');
    title('red-uniform CDF, blue-ecdf');
%     disp(length(cur_hist
    title([num2str( unif_ach_idx( cur_hist ))]);     
end
%% ach_risetime
figure();
for j=1:3
    subplot(1,3,j);
    cur_hist = hist{g(j)};
    idx = abs(cur_hist(:,1))<=30;
    upsample_hist = (fft_upsample(cur_hist(idx,2), 8 )');
    upsample_hist = upsample_hist.*(upsample_hist>=0);
    upsample_times = cur_hist(idx,1);
    upsample_times = linspace(upsample_times(1),upsample_times(end),length(upsample_hist));
    idx = upsample_times>=0;
    cdf_hist = cumsum(upsample_hist(idx));
    cdf_hist = cdf_hist/max(cdf_hist);
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(upsample_times(idx),cdf_hist/max(cdf_hist),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(upsample_times(idx),cdf_hist/max(cdf_hist), 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
    hold on;
    rgb = [ 46 204 5 ]./255;
    factor = 1/exp(1);
    plot(upsample_times(idx),(1/exp(1))*max(cdf_hist)*(cdf_hist<=factor*max(cdf_hist)),'color',rgb, 'LineWidth',2); 
    hold off;
    axis tight;
    grid on;
        xlabel('\tau [ms]');
    ylabel('CDF');
    % time it takes to gain e
    title([num2str( ach_risetime( cur_hist ) )]);
end

%% MID RANGE ACH
figure();
plot_num =  1;
for j=1:3
    subplot(1,3,j);
    cur_hist = hist{g(j)};
    mean_hist = (conv(cur_hist(:,2)', hann(151)./sum(hann(151)),'same'));
    subplot(1,3,j);
    if(red(g(j)))
        rgb=[255 0 0]./255;
        bar(cur_hist(:,1),mean_hist,'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;   
    else
        rgb=[0 0 255]./255;
        bar(cur_hist(:,1),mean_hist, 'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
        axis tight;
    end
        xlabel('\tau [ms]');
    ylabel('count');
    plot_num = plot_num + 1;
end
%% JUMPING ACH MIDBAND FEATURE
figure();
plot_num =  1;
for j=1:3
   subplot(3,2,2*(j-1)+1);
    cur_hist = hist{g(j)};
    w = 151;
    smooth_hist = (conv(cur_hist(:,2)', hann(w)./sum(hann(w)),'same'));
    idx = cur_hist(:,1)>50 & cur_hist(:,1)<1200;
    smooth_hist = smooth_hist(idx);
    arr = smooth_hist-linspace(smooth_hist(1),smooth_hist(end),length(smooth_hist));
    hold on;
    if(red(g(j)))
       rgb=[255 0 0]./255;
    else
       rgb=[0 0 255]./255;
    end
    bar(cur_hist(idx,1),smooth_hist,'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);  
    plot(cur_hist(idx,1),linspace(smooth_hist(1),smooth_hist(end),length(smooth_hist)),'g', 'LineWidth',2);  
    hold off;
    title('mid band of ACH');
    subplot(3,2,2*(j-1)+2);
    plot(cur_hist(idx,1),arr, '', 'LineWidth',2);  
    hold off;
    disp([]);
    title([num2str(  ach_jmp_idx(cur_hist))]);
        xlabel('\tau [ms]');
    ylabel('count');
    axis tight;
    plot_num = plot_num + 1;
end
%%
figure();
for j=1:3
    cur_hist = hist{g(j)};
    idx = abs(cur_hist(:,1))<=20000;
    w = 81;
    us_factor = 8;
    upsample_hist = (fft_upsample(cur_hist(idx,2), us_factor )');
    upsample_hist = upsample_hist.*(upsample_hist>=0);
    upsample_times = cur_hist(idx,1);
    upsample_times = linspace(upsample_times(1),upsample_times(end),length(upsample_hist));
    smooth_hist = (conv(upsample_hist, hann(w)./sum(hann(w)),'same'));
%     f_sample is 20kh
    f= linspace(-10000,10000,length(upsample_times));
    if(red(g(j)))
       rgb=[255 0 0]./255;
    else
       rgb=[0 0 255]./255;
    end
    psd = (abs(fftshift(fft(smooth_hist))));
    psd = psd;%./sum(psd);
    w = 91;
    smooth_psd = (conv(psd, hann(w)./sum(hann(w)),'same'));
    
    subplot(3,2,2*(j-1)+1);
    bar(f,smooth_psd,'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
    axis([-500 500 -0.0001 100000]);
    grid on;
    title(['PSD  \chi=',num2str( pds_feature_1( cur_hist ) ),'[Hz]']);
    xlabel('frequecny');
    ylabel('arbitrary');
    subplot(3,2,2*(j-1)+2);
    bar(abs(diff(smooth_psd(f>=0))),'EdgeColor',rgb,'FaceColor',rgb ,'BarWidth',1);
    f_n = f(f>=0);
    grid on;
    title(['|(dPSD/df)|  \chi=',num2str( pds_feature_2( cur_hist ) ),'[Hz]']);
    axis([-2 1000 -0.0001 2000]);
        xlabel('frequecny');
    ylabel('arbitrary');
    plot_num = plot_num + 1;
end
