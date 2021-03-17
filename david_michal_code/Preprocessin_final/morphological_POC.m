%%
clear;
close all;
%%
sDavid = load('sDavid');
sDavid = sDavid.sDavid;

%%
time = linspace(-0.8,0.8,256);
mean = sDavid.mean;
% hist =
exc = sDavid.exc;
act = sDavid.act;
inh = sDavid.inh;
% scatter(trouf2peak,fwhm_array)
blue = act | inh;
red = exc==1;
untagged = ~red &~blue;

%%
% 20kHz. 32 samples. width of the spike 1.6ms  
figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    hold on;
    w = (conv(mean{j}, hann(5)./sum(hann(5)),'same'));
%     w = w./sum(w.^2); % normlize so that the energy would be 1
    w = w*(4000/(2^15));
    if ~untagged(j)
        if(red(j))
            plot(time,w ,'r', 'LineWidth',2);
            axis tight;  
        else
            plot(time,w ,'b', 'LineWidth',2);
            axis tight;
        end    
%         title(['mean example #',num2str(plot_num)]);
    title([sDavid.filename{(j)},' ',num2str(sDavid.shankclu((j),1)),' ',...
                num2str(sDavid.shankclu((j),2))], 'Interpreter', 'none');
        grid on;
        ylabel('volatge [\muV]');
        xlabel('time [ms]');
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
        grid off;
        hold off;
        plot_num = plot_num + 1;
    end

end

%% FWHM
figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    hold on;
    w = mean{j}*(4000/(2^15));%(conv(mean{j}, hann(5)./sum(hann(5)),'same'));
    minVal = min(w);
    minInd = max(find(w == minVal));
    fwhm_val = minVal + 0.5*peak2peak(w);
    index1 = find(w <= fwhm_val, 1, 'first');
    index2 = find(w <= fwhm_val, 1, 'last');
    wNew = w(minInd:size(w,2));
    maxVal = max(wNew);
    maxInd = min(find(w == maxVal));%% we added max
    wTP = maxInd - minInd;
    if ~untagged(j)
        if(red(j))
            plot(time,w ,'r', 'LineWidth',2);
            axis tight;  
        else
            plot(time,w ,'b', 'LineWidth',2);
            axis tight;
        end    
        title([num2str(get_fwhm(w))]);
        rgb = [ 46 204 5 ]./255;
        plot(time(index1:1:index2), fwhm_val*ones(1,length(index1:1:index2)),'color',rgb, 'LineWidth',2);
        hold off;
        grid on;
        ylabel('volatge [\muV]');
        xlabel('time [ms]');
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
        grid off;
        hold off;
        plot_num = plot_num + 1;
    end

end

%% t2p
figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    hold on;
    w = mean{j}*(4000/(2^15));%(conv(mean{j}, hann(5)./sum(hann(5)),'same'));
    minVal = min(w);
    minInd = max(find(w == minVal));
    fwhm_val = minVal + 0.5*peak2peak(w);
    index1 = find(w <= fwhm_val, 1, 'first');
    index2 = find(w <= fwhm_val, 1, 'last');
    wNew = w(minInd:size(w,2));
    maxVal = max(wNew);
    maxInd = min(find(w == maxVal));%% we added max
    wTP = maxInd - minInd;
    if ~untagged(j)
        if(red(j))
            plot(time,w ,'r', 'LineWidth',2);
            axis tight;  
        else
            plot(time,w ,'b', 'LineWidth',2);
            axis tight;
        end    
        title([num2str(get_wTP(w'))]);
        rgb = [ 46 204 5 ]./255;
        plot(time(minInd:1:maxInd), linspace(minVal,maxVal,length(minInd:1:maxInd)),'color',rgb, 'LineWidth',2);
%         plot(time(index1:1:index2), fwhm_val*ones(1,length(index1:1:index2)),'color',rgb, 'LineWidth',2);

        grid on;
  ylabel('volatge [\muV]');
        xlabel('time [ms]');
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
        grid off;
        plot_num = plot_num + 1;
    end

end

%% RISE COEFF PART 1

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    hold on;
    w = (conv(mean{j}, hann(5)./sum(hann(5)),'same'))*(4000/(2^15));
    if ~untagged(j)
        if(red(j))
            plot(time,w ,'r', 'LineWidth',2);
            axis tight;  
        else
            plot(time,w ,'b', 'LineWidth',2);
            axis tight;
        end    
        rgb = [ 46 204 5 ]./255;
        minVal = min(w);
        minInd = max(find(w == minVal));
        plot(time(minInd:end), linspace(minVal,w(end),length(time(minInd:1:end))),'color',rgb, 'LineWidth',2);
        title([num2str(rise_coeff(mean{j}))]);
        grid on;
        ylabel('volatge [\muV]');
        xlabel('time [ms]');
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
        grid off;
        plot_num = plot_num + 1;
    end

end

%% RISE COEFF PART 2
figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    hold on;
    w = (conv(mean{j}, hann(5)./sum(hann(5)),'same'))*(4000/(2^15));
    y = w(128:256) -linspace(w(128),w(256),length(128:256));
    if ~untagged(j)
        if(red(j))
            plot(time(128:256),y,'r', 'LineWidth',2);
            axis tight;   
        else
            plot(time(128:256),y ,'b', 'LineWidth',2);
            axis tight;
        end
        title([num2str(rise_coeff(mean{j}))]);
        grid on;
ylabel('volatge [\muV]');
        xlabel('time [ms]');
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
        grid off;
        plot_num = plot_num + 1;
    end
end

%%  first derivative 

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'))*(4000/(2^15));
    w= diff(w);
    hold on;
    if ~untagged(j)
        if(red(j))
            plot(time(2:end),w,'r', 'LineWidth',2);
            axis tight;                  
        else
            plot(time(2:end),w,'b', 'LineWidth',2);
            axis tight;
        end
        grid on;
        ylabel('dV/dt [\muV/ms]');
        xlabel('time [ms]');
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
        grid off;
        title(['first derivative #',num2str(plot_num)]);
        plot_num = plot_num + 1;
    end
end
%%  max_speed 1

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'))*(4000/(2^15));
    w= diff(w);
    hold on;
    if ~untagged(j)
        if(red(j))
            plot(time(2:end),w,'r', 'LineWidth',2);
            axis tight;                  
        else
            plot(time(2:end),w,'b', 'LineWidth',2);
            axis tight;
        end
        roi = 131:200;
        ind = find(w>w(131));
        ind = ind(ind>131);
        plot(time(ind),w(ind),'m', 'LineWidth',2);
        grid on;
        ylabel('dV/dt [\muV/ms]');
        xlabel('time [ms]');
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
        grid off;
        title([num2str(max_speed(mean{j}))]);
        plot_num = plot_num + 1;
    end
end

%%  max_speed 2

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'));
    w= diff(w);
    hold on;
    if ~untagged(j)
        
        roi = 131:200;
        ind = find(w>w(131));
        ind = ind(ind>131);
        w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'))*(4000/(2^15));
        if(red(j))
            plot(time,w,'r', 'LineWidth',2);
            axis tight;                  
        else
            plot(time,w,'b', 'LineWidth',2);
            axis tight;
        end

        plot(time(ind),w(ind),'m', 'LineWidth',2);
        grid on;
        ylabel('voltage [\muV]');
        xlabel('time [ms]');
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
        grid off;
        title([num2str(max_speed(mean{j}))]);
        plot_num = plot_num + 1;
    end
end
%% second derivative 

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'))*(4000/(2^15));
    w = diff(diff(w(20:end-10)));
    w = (conv(w, hann(5)./sum(hann(5)),'same'));
    hold on;
    if ~untagged(j)
        if(red(j))
            plot(time(23:end-9),w,'r', 'LineWidth',2);
            axis tight;                  
        else
            plot(time(23:end-9),w,'b', 'LineWidth',2);
            axis tight;
        end
    roi = (65:95);
%     plot(roi, w(roi),'m', 'LineWidth',2);
        title(['second derivative #',num2str(plot_num)]);
        grid on;
        ylabel('d^2V/d^2t [V/s^2]');
        xlabel('time [ms]');
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
        grid off;
    plot_num = plot_num + 1;
    end
end
%% BREAK MEASURE POC

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'));
    w = w(20:end-10)*(4000/(2^15));
    w = (conv(w, hann(5)./sum(hann(5)),'same'));
    hold on;
    if ~untagged(j)
        if(red(j))
            plot(time(20:end-10),w,'r', 'LineWidth',2);
            axis tight;                  
        else
            plot(time(20:end-10),w,'b', 'LineWidth',2);
            axis tight;
        end
    roi = (60:95);
    plot(time(roi+20), w(roi),'m', 'LineWidth',2);
    title([num2str(break_measure(mean{j}))]);
      ylabel('voltage [\muV]');
        xlabel('time [ms]');
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
        grid off;
    plot_num = plot_num + 1;
    end
end

%% BREAK MEASURE POC 2

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'));
    w = diff(diff(w(20:end-10)));
    w = (conv(w, hann(5)./sum(hann(5)),'same'));
    hold on;
    if ~untagged(j)
        if(red(j))
            plot(time(23:end-9),w,'r', 'LineWidth',2);
            axis tight;                  
        else
            plot(time(23:end-9),w,'b', 'LineWidth',2);
            axis tight;
        end
    roi = (65:95);
    plot(time(roi+23), w(roi),'m', 'LineWidth',2);
    title([num2str(break_measure(mean{j}))]);
        ylabel('d^2V/d^2t [V/s^2]');
        xlabel('time [ms]');
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
        grid off;
        hold off;
    plot_num = plot_num + 1;
    end
end

%% GET ACC POC

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'))*(4000/(2^15));
    w = diff(diff(w));
    w = w(10:end-10);
    w = (conv(w, hann(15)./sum(hann(15)),'same'));
    hold on;
    if ~untagged(j)
        if(red(j))
            plot(time(13:end-9),w,'r', 'LineWidth',2);
            axis tight;
        else
            plot(time(13:end-9),w,'b', 'LineWidth',2);
            axis tight;
        end
        roi = (128:170);
        plot(time(roi+11), w(roi),'m', 'LineWidth',2);
        title([num2str(get_acc(mean{j}))]);
         grid on;
      ylabel('d^2V/d^2t [V/s^2]');
        xlabel('time [ms]');
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
        grid off;
        plot_num = plot_num + 1;
    end
end

%% GET ACC POC 2 
figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'))*(4000/(2^15));
    w = diff(diff(w));
    w = w(10:end-10);
    w = (conv(w, hann(15)./sum(hann(15)),'same'));
    hold on;
    if ~untagged(j)
        
        roi = (128:170);

        title([num2str(get_acc(mean{j}))]);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'))/1000;
        if(red(j))
            plot(time,w,'r', 'LineWidth',2);
            axis tight;
        else
            plot(time,w,'b', 'LineWidth',2);
            axis tight;
        end
                plot(time(roi), w(roi),'m', 'LineWidth',2);
         grid on;
ylabel('volatge [\muV]');
        xlabel('time [ms]');
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
        grid off;
        plot_num = plot_num + 1;
    end
end

%% SMILE OR CRY POC 1

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'))*(4000/(2^15));
    w = (w);
    w = w(10:end-10);
    hold on;
    if ~untagged(j)
        if(red(j))
            plot(time(10:end-10),w,'r', 'LineWidth',2);
            axis tight;
        else
            plot(time(10:end-10),w,'b', 'LineWidth',2);
            axis tight;
        end
        roi = (170:230);
        plot(time(roi+10), w(roi),'m', 'LineWidth',2);%0*ones(1,length(roi)),'m', 'LineWidth',2);
        title([num2str(smile_or_cry(mean{j}))]);
ylabel('volatge [\muV]');
        xlabel('time [ms]');
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
        grid off;
        hold off;
        plot_num = plot_num + 1;
    end
end
%% SMILE OR CRY POC 2

figure();
plot_num = 1;
for j = 1:27
    subplot(3,4,plot_num);
    w = (conv(mean{j}, hann(15)./sum(hann(15)),'same'));
    w = (diff(w));
    w = w(10:end-10);
    hold on;
    if ~untagged(j)
        if(red(j))
            plot(time(12:end-9),w,'r', 'LineWidth',2);
            axis tight;
        else
            plot(time(12:end-9),w,'b', 'LineWidth',2);
            axis tight;
        end
        roi = (170:230);
        plot(time(roi+10), w(roi),'m', 'LineWidth',2);%0*ones(1,length(roi)),'m', 'LineWidth',2);
        title(['\chi = ',num2str(smile_or_cry(mean{j}))]);
       grid on;
        ylabel('dV/dt [\muV/ms]');
        xlabel('time [ms]');
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
        grid off;
        hold off;
        plot_num = plot_num + 1;
    end
end

