%%  Test Implementation of GMM-EM Algorithm
%    on 2D Datasets. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Load 2D GMM-EM Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear variables; close all; clc;

addpath(genpath('../check_utils'))
addpath(genpath('../../ML_toolbox-master'))
addpath(genpath('../../TP2-KMeans-Assignment'))

%% (1a) Load 2D data for testing Gausspdf and Covariance Matrices
load('../../TP4-GMM-Datasets/2D-Gaussian.mat')

% Visualize Dataset
% Add path

options.class_names = {};
options.title       = '2D Gaussian Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;

%% (1b) Load 2D dataset for testing GMM-EM & Likelihood
load('../../TP4-GMM-Datasets/2D-GMM.mat')

% Visualize Dataset
options.labels      = labels;
options.class_names = {};
options.title       = '2D GMM Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(4);
ml_plot_centroid(gmm.Mu',colors);hold on; 
ml_plot_gmm_contour(gca,gmm.Priors,gmm.Mu,gmm.Sigma,colors);
grid on; box on;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              2) Check my_gaussPDF.m function               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ATTENTION: Load Datatest (1a) before running this code block! %%%%%%
    
% Real Mu and Sigma used for 1a
Mu = [1;1];
Sigma = [1, 0.5; 0.5, 1];

% Check my_gausspdf.m function
clc;
check_mygaussPDF(X, Mu, Sigma);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             3) Check my_gmmloglik.m function               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ATTENTION: Load Datatest (1b) before running this code block! %%%%%%

% Load gmm parameters with real values
Priors = gmm.Priors;
Mu     = gmm.Mu;
Sigma  = gmm.Sigma;

% Check my_gmmLogLik.m function
clc;
check_mygmmLogLik(X, Priors, Mu, Sigma);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%            4) Check my_covariance.m function               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ATTENTION: Load Datatest (1a) before running this code block! %%%%%%

% Check my_covariance.m function
clc;
check_mycovariance(X); 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       5) Visualize result of my_covariance.m function      %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ATTENTION: Load Datatest (1a) before running this code block! %%%%%%

%%%%% Visualize different covariance matrices %%%%
clc; close all; 

[Sigma_full, Sigma_diag, Sigma_iso] = visualize_covariances(X);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       6) Visualize result of my_gmmInit.m function         %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ATTENTION: Load Datatest (1b) before running this code block! %%%%%%

% Set GMM Hyper-parameters
K = 4; cov_type = 'full'; plot_iter = 1;
clc; close all;

% Run GMM-INIT function, estimates and visualizes initial parameters for EM algorithm
[ Priors0, Mu0, Sigma0 ] = my_gmmInit(X, K, cov_type,  plot_iter);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%        7) Visualize result of my_gmmEM.m function          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%% ATTENTION: Load Datatest (1b) before running this code block! %%%%%%
clc; close all;

% Set GMM Hyper-parameters
K = 4; cov_type = 'iso';  plot_iter = 1;
%[Priors, Mu, Sigma] = my_gmmInit(X, K, cov_type, plot_iter);
%save('InitialVal.mat', 'Priors', 'Mu', 'Sigma')
    
%%%% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
tic;
[Priors, Mu, Sigma] = my_gmmEM(X, K, cov_type,  plot_iter);
toc;


% Visualize GMM pdf from learnt parameters
ml_plot_gmm_pdf(X, Priors, Mu, Sigma)