%%  Test Implementation of GMR Algorithm
%   On some datasets
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              1) Load 1D Dataset                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('../../ML_toolbox-master'))
addpath(genpath('../check_utils'))
addpath(genpath('../Part1'))
addpath(genpath('../Prev-TP-Solutions'))
%addpath(genpath('../../TP2-KMeans-Assignment'))
%addpath(genpath('../../TP3-KNN-Assignment'))
%addpath(genpath('../../TP4-GMM-Assignment'))
%% %%%  (1a) Generate Non-Linear Data from sine function %%%%%%
clc; clear all; close all; dataset_type = '1d-sine';
[ X, y_true, y ] = load_regression_datasets( dataset_type );

%% %%% (1b) Generate Non-Linear Data from a sinc function %%%%%%
clc; clear all; close all; dataset_type = '1d-sinc';
[ X, y_true, y ] = load_regression_datasets( dataset_type );

%% %%%% (1c) Non-linear high dimensional Dataset %%%%%%
clc; clear all; close all;
%% %%%%%%%%%%%%%%%%%% Load 11-D Wine Quality Dataset  %%%%%%%%%%%%%%%%%%%%%
% This Wine Dataset uses chemical analysis determine the origin of wines
% https://archive.ics.uci.edu/ml/datasets/Wine
raw_data = table2array(ml_load_data('winequality-white.csv','csv'));
X_raw = raw_data(:,1:11);
y = raw_data(:,12);
M = size(X_raw,2);

% Perform PCA on the data
[V, L, pca_Mu] = my_pca(X_raw');

% Find the number of dimensions to project from the eigen values
p = 2;

% Project Data to Choosen Principal Components
[A_p, X_proj] = project_pca(X_raw', pca_Mu, V, p);

X = X_proj';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        2) Fit a GMM Model to your regressive data       %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all;

% Define X for gmm Estimation
X_total = [X';y'];

% Estimate Mu, Sigma & Priors
K = 4;
cov_type ='full'; plot_iter = 1;

[Priors, Mu, Sigma] = my_gmmEM(X_total, K, cov_type, plot_iter);

if size(X_total,1) == 2
    ml_plot_gmm_pdf(X_total, Priors, Mu, Sigma)
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    3) Compute Regressive Signal and Variance from GMM     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; 

% Compute Regressive signal and variance
N = size(X,2); P = size(y,2);
in  = 1:N;       % input dimensions
out = N+1:(N+P); % output dimensions
[y_est, var_est] = my_gmr(Priors, Mu, Sigma, X', in, out);

%% Visualize Regression Result (for 1D Datasets only 1a/1b)
% Plot Datapoints
options             = [];
options.points_size = 15;
options.title       = 'Estimated y=f(x) from Gaussian Mixture Regression';
options.labels      = zeros(length(y_est),1);
if exist('h4','var') && isvalid(h4), delete(h4);end
h4 = ml_plot_data([X,y],options); hold on;

% Plot True function 
plot(X,y_true,'-k','LineWidth',1); hold on;

% Plot Estimated function 
options             = [];
options.var_scale   = 2;
options.title       = 'Estimated y=f(x) from Gaussian Mixture Regression';
options.plot_figure = false;
ml_plot_gmr_function(X, y_est, var_est, options)
legend({'data','y = f(x)','$Var\{p(y|x)\}$','$+2\sigma\{p(y|x)\}$', ...
    '$-2\sigma\{p(y|x)\}$','$\hat{y} = E\{p(y|x)\}$' }, 'Interpreter','latex')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               4) Compute regression metrics                %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check regression metrics
clc;
check_regression_metrics(y_est,y');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      5) Cross validation (cross_validation_gmr.m)          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM parameters
cov_type = 'full';  plot_iter = 0;

% Cross-validation parameters
tt_ratio  = 0.7;    % train/test ratio
k_range   = 1:10;   % range of K to evaluate
F_fold    = 10;     % # of Folds for cv

% Compute F-fold cross-validation
[MSE_F_fold, NMSE_F_fold, R2_F_fold, AIC_F_fold, BIC_F_fold, std_MSE_F_fold, std_NMSE_F_fold, std_R2_F_fold, ...,
    std_AIC_F_fold, std_BIC_F_fold] = cross_validation_gmr(X', y', cov_type, plot_iter, F_fold, tt_ratio, k_range);

%% Plot GMM Model Selection Metrics for F-fold cross-validation with std
figure;
errorbar(k_range',AIC_F_fold(k_range)', std_AIC_F_fold(k_range)','--or','LineWidth',2); hold on;
errorbar(k_range',BIC_F_fold(k_range)', std_BIC_F_fold(k_range)','--ob','LineWidth',2);
grid on
xlabel('Number of K components'); ylabel('AIC/BIC Score')
legend('AIC', 'BIC')

%% Plot Regression Metrics for F-fold cross-validation with std
figure;
[ax,hline1,hline2]=plotyy(k_range',MSE_F_fold(k_range)',[k_range' k_range'],[NMSE_F_fold(k_range)' R2_F_fold(k_range)']);
delete(hline1);
delete(hline2);
hold(ax(1),'on');
errorbar(ax(1),k_range', MSE_F_fold(k_range)', std_MSE_F_fold(k_range)','--o','LineWidth',2,'Color', [0 0.447 0.741]);
hold(ax(2),'on');
errorbar(ax(2),k_range',NMSE_F_fold(k_range)', std_NMSE_F_fold(k_range)','--or','LineWidth',2);
errorbar(ax(2),k_range',R2_F_fold(k_range)', std_R2_F_fold(k_range)','--og','LineWidth',2);
xlabel('Number of K components'); ylabel('Measures')
legend('MSE', 'NMSE', 'Rsquared')
grid on
