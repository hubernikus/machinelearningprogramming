function [ MSE_F_fold, NMSE_F_fold, R2_F_fold, AIC_F_fold, BIC_F_fold, std_MSE_F_fold, ...,
    std_NMSE_F_fold, std_R2_F_fold, std_AIC_F_fold, std_BIC_F_fold] = cross_validation_gmr( X, y, ...,
    cov_type, plot_iter, F_fold, tt_ratio, k_range )
%CROSS_VALIDATION_REGRESSION Implementation of F-fold cross-validation for regression algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y         : (P x M) array representing the y vector assigned to
%                           each datapoints
%       o F_fold    : (int), the number of folds of cross-validation to compute.
%       o tt_ratio  : (double), Training/Testing Ratio.
%       o k_range   : (1 x K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------
%
%       o MSE_F_fold      : (1 x K), Mean Squared Error computed for each value of k averaged over the number of folds.
%       o NMSE_F_fold     : (1 x K), Normalized Mean Squared Error computed for each value of k averaged over the number of folds.
%       o R2_F_fold       : (1 x K), Coefficient of Determination computed for each value of k averaged over the number of folds.
%       o AIC_F_fold      : (1 x K), Mean AIC Scores computed for each value of k averaged over the number of folds.
%       o BIC_F_fold      : (1 x K), Mean BIC Scores computed for each value of k averaged over the number of folds.
%       o std_MSE_F_fold  : (1 x K), Standard Deviation of Mean Squared Error computed for each value of k.
%       o std_NMSE_F_fold : (1 x K), Standard Deviation of Normalized Mean Squared Error computed for each value of k.
%       o std_R2_F_fold   : (1 x K), Standard Deviation of Coefficient of Determination computed for each value of k averaged over the number of folds.
%       o std_AIC_F_fold  : (1 x K), Standard Deviation of AIC Scores computed for each value of k averaged over the number of folds.
%       o std_BIC_F_fold  : (1 x K), Standard Deviation of BIC Scores computed for each value of k averaged over the number of folds.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define Dimensions
[N, M] = size(X);
[P,~] = size(y);
K = length(k_range);

% Define Dimension Parameters
in = [1:N];
out = [N+1:N+P];

% % Define emty vectors % % 
% Mean valuesvalues
MSE_F_fold = zeros(1,K); NMSE_F_fold = zeros(1,K); R2_F_fold = zeros(1,K);
AIC_F_fold = zeros(1,K); BIC_F_fold = zeros(1,K); 
% Standart Deviation
std_MSE_F_fold = zeros(1,K); std_NMSE_F_fold = zeros(1,K); std_R2_F_fold = zeros(1,K);
std_AIC_F_fold = zeros(1,K); std_BIC_F_fold = zeros(1,K);
% Temporary safe for value of evaluation
MSE_temp = zeros(1,F_fold); NMSE_temp = zeros(1,F_fold); R2_temp = zeros(1,F_fold);
AIC_temp = zeros(1,F_fold); BIC_temp = zeros(1,F_fold);


% Estimation of Regression & AIC/BIC for each k in K-range, F_fold times
for kk = 1:K 
    for ff = 1:F_fold 
        fprintf('K=%d; Fold=%d/%d \n', k_range(kk),ff,F_fold);
        
        % Split Data to train/test
        [X_train, y_train, X_test, y_test] = split_data(X, y, tt_ratio);
        
        % Regression Metrics 
        [Priors, Mu, Sigma] = my_gmmEM([X_train;y_train], k_range(kk), cov_type, plot_iter);
        [y_est, var_est] = my_gmr(Priors, Mu, Sigma, X_test, in, out);
        [MSE_temp(ff), NMSE_temp(ff), R2_temp(ff)] = my_regression_metrics(y_est, y_test);
        
        %
        %[~, Mu, Sigma] = my_gmmEM([X_train, k_range(kk), cov_type, plot_iter);
        [AIC_temp(ff), BIC_temp(ff)] = gmm_metrics([X_train;y_train], Priors, Mu, Sigma, cov_type);  %?? test/train
    end
    % Calculate mean and std for each Fold and assign to list 
    MSE_F_fold(kk) = mean(MSE_temp); std_MSE_F_fold(kk) = std(MSE_temp);
    NMSE_F_fold(kk) = mean(NMSE_temp); std_NMSE_F_fold(kk) = std(NMSE_temp);
    R2_F_fold(kk) = mean(R2_temp); std_R2_F_fold(kk) = std(R2_temp);
    
    AIC_F_fold(kk) = mean(AIC_temp); std_AIC_F_fold(kk) = std(AIC_temp);
    BIC_F_fold(kk) = mean(BIC_temp); std_BIC_F_fold(kk) = std(BIC_temp);
end

end

