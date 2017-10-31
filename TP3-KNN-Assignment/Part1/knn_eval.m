function [ ] = knn_eval( X_train, y_train, X_test, y_test, k_range )
%KNN_EVAL Implementation of kNN evaluation.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%       o k_range  : (1 X K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assign variables
[N, M_train] = size(X_train);
[~,M_test] = size(X_test);

% Type of the distance measurement
type = 'L2';
    
% Calculate the accuracies over the range of k
acc = zeros(1, length(k_range));
for i = 1:length(k_range)
    y_est = my_knn(X_train, y_train, X_test, k_range(i), type);
    acc(i) = my_accuracy(y_test, y_est);    
end


% Create plot    
figure('Name', 'Classification Evaluation for KNN', 'Color', [1 1 1]);
plot(k_range,acc, 'r--o'); hold on; grid on;
title('Clafficiation Evaluation for KNN')
xlabel('k'); ylabel('Acc');


end

