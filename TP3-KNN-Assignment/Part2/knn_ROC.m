function [ TP_rate, FP_rate ] = knn_ROC( X_train, y_train, X_test, y_test, k_range )
%KNN_ROC Implementation of ROC curve for kNN algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%       o k_range  : (1 x K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------

%       o TP_rate  : (1 x K), True Positive Rate computed for each value of k.
%       o FP_rate  : (1 x K), False Positive Rate computed for each value of k.
%        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Populate True Positive Rate and True Negative Rate vectors (for each k-value)

% Define Dimensions
[N, M_train] = size(X_train);
[~,M_test] = size(X_test);
K = length(k_range);

% True positive Rate (TPR) // sensitivity/recall
TP_rate = zeros(1,K);
% False positive Rate (FPR) 
FP_rate = zeros(1,K);

% Type of the distance measurement
type = 'L2';

% Calculate Confusion with C = [TP FP; FN TN]
for i = 1:K
    y_est = my_knn(X_train, y_train, X_test, k_range(i), type);
    C = confusion_matrix(y_test, y_est);
    TP_rate(i) = C(1,1)/(C(1,1)+C(1,2));
    FP_rate(i) = C(2,1)/(C(2,1)+C(2,2));
end
    

%figure('Name', 'ROC curbe for KNN', 'Color', [1 1 1])
%cellstr(['k = ', num2str(k_range)]); % Label the points
%plot(FP_rate,TP_rate,'--o', 'Color', [1 0 0] ); 
%hold on; grid on; 
%title('ROC curbe for KNN')
%xlabel('False Positive rate'); ylabel('True Positive rate');

end
