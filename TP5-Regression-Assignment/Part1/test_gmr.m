%%  Test Implementation of Gaussian Mixture Regressin Algorithm on 1D/2D Datasets
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               1) Load 1D Regression Datasets               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('../../ML_toolbox-master'))
addpath(genpath('../check_utils'))

addpath(genpath('../../TP2-KMeans-Assignment'))
%addpath(genpath('../../TP3-KNN-Assignment'))
addpath(genpath('../../TP4-GMM-Assignment'))
%% %%%  (1a) Generate Data from a noisy line %%%%%%
clc; clear all;dataset_type = '1d-linear';
[ X, y_true, y ] = load_regression_datasets( dataset_type );

%% %%%  (1b) Generate Non-Linear Data from sine function %%%%%%
clc; clear all; dataset_type = '1d-sine';
[ X, y_true, y ] = load_regression_datasets( dataset_type );

%% %%% (1c) Generate Non-Linear Data from a sinc function %%%%%%
clc; clear all;dataset_type = '1d-sinc';
[ X, y_true, y ] = load_regression_datasets( dataset_type );

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             2) Load 2D Regression Datasets                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%% (2a) Sin/Cos Dataset %%%%%%
clc; clear all; close all;
dataset_type = '2d-cossine';
[ X, y_true, y ] = load_regression_datasets( dataset_type );

%% %%%% (2b) Monotonically Decreasing Dataset %%%%%%
clc; clear all; close all;
dataset_type = '2d-mono';
[ X, y_true, y ] = load_regression_datasets( dataset_type );

%% %%%% (2c) Non-linear 2D GMM Dataset %%%%%%
clc; clear all; close all;
dataset_type = '2d-gmm';
[ X, y_true, y ] = load_regression_datasets( dataset_type );

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    3) Learn the GMM Model from your regression data       %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all;
% %%% TASK 1) FILL THIS CODE BLOCK!
%x = X;
X_val = [X';y'];

% Estimate Mu, Sigma & Priors
K = 5;
cov_type = 'full'; % ['full','diag','iso'] choose from
plot_iter = 1; % Show plot after iteratNion
[Priors, Mu, Sigma ] = my_gmmEM(X_val, K, cov_type, plot_iter);

if size(X_val,1) ==2
    ml_plot_gmm_pdf(X_val, Priors, Mu, Sigma)
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         4) Check my_gmr.m function             %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;
% Compute Expectation of conditional E{p(Xi_y|Xi_x,\theta)}
% and variance of conditional Var{p(Xi_y|Xi_x,\theta)}

% Check your implementation of Gaussian Mixture Regression
N = size(X,2); P = size(y,2); 
in  = 1:N;  
out = N+1:(N+P);
clc;
check_mygmr(X, Priors, Mu, Sigma, in, out); 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    5) Validate my_gmr.m function on 1D Dataset    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all;
% Compute Regressive signal and variance
N = size(X,2); P = size(y,2);
in  = 1:N;       % input dimensions
out = N+1:(N+P); % output dimensions
[y_est, var_est] = my_gmr(Priors, Mu, Sigma, X', in, out);

% Plot Datapoints
options             = [];
options.points_size = 15;
options.title       = 'Estimated y=f(x) from Gaussian Mixture Regression';
options.labels      = zeros(length(y_est),1);
if exist('h4','var') && isvalid(h4), delete(h4);end
h4 = ml_plot_data([X(:),y(:)],options); hold on;

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


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    6) Validate my_gmr.m function on 2D Dataset    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute Regressive signal and variance
N = size(X,2); P = size(y,2); M = size(X,1);
in  = 1:N;       % input dimensions
out = N+1:(N+P); % output dimensions
[y_est, var_est] = my_gmr(Priors, Mu, Sigma, X', in, out);

% Function handle for my_gmr.m
f = @(X) my_gmr(Priors,Mu,Sigma,X, in, out);

% Plotting Options for Regressive Function
options           = [];
options.title     = 'Estimated y=f(x) from Gaussian Mixture Regression';
options.regr_type = 'GMR';
options.surf_type = 'surf';
ml_plot_value_func(X,f,[1 2],options);hold on

% Plot Training Data
options = [];
options.plot_figure = true;
options.points_size = 12;
options.labels = zeros(M,1);
options.plot_labels = {'x_1','x_2','y'};
ml_plot_data([X y],options);

