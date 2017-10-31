%%  Test Implementation of Linear Regression Algorithm on 1D/2D Datasets
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               1) Load 1D Regression Datasets               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('../../ML_toolbox-master'))
addpath(genpath('../check_utils'))
%% %%%  (1a) Generate Data from a noisy line %%%%%%
clc; clear all; dataset_type = '1d-linear';
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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        3) Check my_lr.m function      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check your implementation of linear regression
% Try testing with 1D and 2D Datasets
clc;
check_mylr(X,y); 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        4) Validate my_lr.m function on 1D Dataset    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ==> Load one of the 1D Datasets from code block 1a)/1b)/1c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Apply Linear Regression
[w, y_est] = my_lr(X,y);

% Plot Datapoints
options             = [];
options.points_size = 15;
options.title       = 'Estimated y=f(x) from Linear Regression';
options.labels      = zeros(length(y_est),1);
if exist('h3','var') && isvalid(h3), delete(h3);end
h3 = ml_plot_data([X(:),y(:)],options); hold on;

% Plot True function 
plot(X,y_true,'-k','LineWidth',2); hold on;

% Plot Estimated function 
plot(X,y_est,'-r','LineWidth', 2); hold on;
legend({'data','y = f(x)', '$\hat{y} = \hat{f}$(x)'}, 'Interpreter','latex')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        5) Validate my_lr.m function on 2D Dataset    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ==> Load one of the 2D Datasets from code block 2a)/2b)/2c)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Apply Linear Regression to 2D dataset %%%%%%
[w, y_est] = my_lr(X,y);

% Function handle for my_lr.m
f = @(X)(w'*X');
M = length(X);

% Plotting Options for Regressive Function
options           = [];
options.title     = 'Estimated y=f(x) from Linear Regression';
options.surf_type = 'surf';
options.type      = 'gmr';
ml_plot_value_func(X,f,[1 2],options);hold on

% Plot Training Data
options = [];
options.plot_figure = true;
options.points_size = 12;
options.labels = zeros(M,1);
options.plot_labels = {'$x_1$','$x_2$','y'};
ml_plot_data([X y],options);

