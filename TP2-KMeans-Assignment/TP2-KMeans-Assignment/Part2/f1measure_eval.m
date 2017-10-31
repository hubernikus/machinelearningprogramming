function [] =  f1measure_eval(X, K_range,  repeats, init, type, MaxIter, true_labels)
%F1MEASURE_EVAL Implementation of the k-means evaluation with F1-Measure
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


f1measure_curve = zeros(1, length(K_range));

% Populate Curves
for i=1:length(K_range)
    
    % Select K from K_range
    K = K_range(i); 
    
    % Repeat k-means X times
    f1measure_ = zeros(1, repeats);
    for ii = 1:repeats
        plot_iter = 0;  
        [labels, Mu] =  my_kmeans(X, K, init, type, MaxIter, plot_iter);  
        f1measure_(ii)  = my_f1measure(labels', true_labels);
    end 
    
    % Get the mean of those X repeats
    f1measure_curve(i) = mean (f1measure_);
    
end

% Plot Metric Curves
if exist('h_metrics','var') && isvalid(h_metrics),  delete(h_metrics); end
h_metrics = figure; hold on;

plot(f1measure_curve,'--o', 'LineWidth', 1); 
xlabel('K')
legend('F1-Measure')
title('Clustering F1-Measure')
grid on



end