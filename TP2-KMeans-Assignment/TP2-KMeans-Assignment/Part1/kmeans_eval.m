function [] =  kmeans_eval(X, K_range,  repeats, init, type, MaxIter)
%KMEANS_EVAL Implementation of the k-means evaluation with clustering
%metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats  : (1 X 1), # times to repeat k-means
%       o K_range  : (1 X K), Range of k-values to evaluate
%       o init     : (string), type of initialization {'random','uniform','plus'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%
%   output ----------------------------------------------------------------
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RSS_curve = zeros(1, length(K_range));
AIC_curve = zeros(1, length(K_range));
BIC_curve = zeros(1, length(K_range));
plot_iter = 0;  
    
% Populate Curves
for i=1:length(K_range)
    
    % Select K from K_range
    K = K_range(i); 
    
    % Repeat k-means X times
    RSS_ = zeros(1, repeats); AIC_ = zeros(1, repeats); BIC_= zeros(1, repeats);     
    for ii = 1:repeats
        
        [labels, Mu] = my_kmeans(X, K, init, type, MaxIter, plot_iter);
        [RSS_(ii),AIC_(ii),BIC_(ii)] = my_metrics(X,labels,Mu);
    end 
    
    % Get the mean of those X repeats <=== IMPLEMENT THIS!
    RSS_curve(i) = mean(RSS_);
    AIC_curve(i) = mean(AIC_);
    BIC_curve(i) = mean(BIC_);
    
    
end

% Plot Metric Curves
if exist('h_metrics','var') && isvalid(h_metrics),  delete(h_metrics); end
h_metrics = figure;hold on;

plot(RSS_curve,'--o', 'LineWidth', 1); hold on;
plot(AIC_curve,'--o', 'LineWidth', 1); hold on;
plot(BIC_curve,'--o', 'LineWidth', 1); hold on;
xlabel('K')
legend('RSS', 'AIC', 'BIC')
title('Clustering Evaluation metrics')
grid on
end