%%  Test Implementation of K-NN (K-Nearest Neighbors)
%    on high dimensional Datasets.
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           1) Load 9D KNN Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Load Wisconsin Breast Cancer Dataset
clear all;
close all;
clc;

% On Windows (classroom pc): Fill in your ML_toolbox_path HERE <====
ml_toolbox_path = '../../ML_toolbox-master/';
[X, y, class_names] = ml_load_data(strcat(ml_toolbox_path,'data/breast-cancer-wisconsin.csv'),'csv','last');

% On Linux Systems
% [X, y, class_names] = ml_load_data('breast-cancer-wisconsin.csv','csv','last');

% Breast-Cancer-Wisconsin Dataset
% https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
% Nine real-valued features are computed for each cell nucleus:
%   Feature                          Range
%   1. Clump Thickness               1 - 10
%   2. Uniformity of Cell Size       1 - 10
%   3. Uniformity of Cell Shape      1 - 10
%   4. Marginal Adhesion             1 - 10
%   5. Single Epithelial Cell Size   1 - 10
%   6. Bare Nuclei                   1 - 10
%   7. Bland Chromatin               1 - 10
%   8. Normal Nucleoli               1 - 10
%   9. Mitoses                       1 - 10

% Transpose matrices to have datapoints as columns and dimensions as rows
X = X';
y_ = y';

% Convert labels to binary 0/1 (positive/negative)
y = zeros(size(y_));
y(y_ == 1) = 0; 
y(y_ == 2) = 1;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     2) Data Handling for Classification (split_data.m)        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Training/Testing Ratio
% ...
tt_ratio = 0.3;

% Split data into a training dataset that kNN can use to make predictions
% and a test dataset that we can use to evaluate the accuracy of the model.
% ...
[X_train, y_train, X_test, y_test] = split_data(X, y, tt_ratio);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         3) Choosing K by visualizing knn_eval.m            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Range of K to test accuracy
% ...
[~,M_train] = size(X_train);
k_range = [1:5:M_train];


% Run knn_eval
% ...
knn_eval(X_train, y_train, X_test, y_test, k_range)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      4) Test Confusion Matrix (confusion_matrix.m)         %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select k
k = 5;

% kNN classification of test set
y_est =  my_knn(X_train, y_train, X_test, k, 'L2');

% Confusion matrix computation for the classified data
C = confusion_matrix(y_test, y_est);

% Comparison with Matlab confusion matrix function
[C_matlab, order] = confusionmat(y_test, y_est, 'order', [0 1]);
if(norm(C-C_matlab) == 0)
    fprintf('[Test Confusion Matrix]: ok\n');
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                5) Plot ROC curve (knn_ROC.m)               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Split data randomly between train and test
tt_ratio = 0.1;
[X_train, y_train, X_test, y_test] = split_data( X, y, tt_ratio );

% Compute ROC curve
k_range = [1:8:ceil(length(y)*tt_ratio)];
[TP_rate, FP_rate] = knn_ROC( X_train, y_train, X_test, y_test, k_range );

% Plot ROC Curve
figure;
plot(FP_rate, TP_rate, '--o', 'LineWidth', 1, 'Color', [1 0 0]); hold on;
xlabel('False Positive rate'); ylabel('True Positive rate')
title('ROC curve for KNN')
grid on
for i = 1:length(k_range)
    current_k = k_range(i);
    text(FP_rate(i)+0.001,TP_rate(i)-0.001+0.001*mod(i,3),['k = ' num2str(current_k)])
end
% hold off

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%          6) Cross validation (cross_validation.m)          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tt_ratio = 0.2;
k_range = [1:8:ceil(length(y)*tt_ratio)];
F_fold = 10;

% Compute F-fold cross-validation
[TP_rate_F_fold, FP_rate_F_fold, std_TP_rate_F_fold, std_FP_rate_F_fold] =  cross_validation(X, y, F_fold, tt_ratio, k_range);

% Plot ROC curve for F-fold cross-validation
figure;
plot(FP_rate_F_fold, TP_rate_F_fold, '--o', 'LineWidth', 2, 'Color', [1 0 0]); hold on;
xlabel('False Positive rate'); ylabel('True Positive rate')
title('ROC curve for KNN')
grid on
for i = 1:length(k_range)
    current_k = k_range(i); 
    text(FP_rate_F_fold(i)+0.015,TP_rate_F_fold(i)-0.0005,['k = ' num2str(current_k)])
end

%% Plot ROC curve with standard deviation
figure
hold on;
yneg = std_TP_rate_F_fold;
ypos = yneg;
xneg = std_FP_rate_F_fold;
xpos = xneg;
plot(FP_rate_F_fold, TP_rate_F_fold, '--o', 'LineWidth', 2, 'Color', [1 0 0]); hold on;
herrorbar(FP_rate_F_fold,TP_rate_F_fold,xpos,xneg,'ko');
errorbar(FP_rate_F_fold,TP_rate_F_fold,yneg,ypos,'ko');
xlabel('False Positive rate'); ylabel('True Poclsitive rate')
title('ROC curve for KNN')
grid on
hold off
