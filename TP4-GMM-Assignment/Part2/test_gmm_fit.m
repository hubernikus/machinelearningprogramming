%%  Test Implementation of GMM-EM Algorithm
%    on 2D Datasets. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Load 2D GMM Fit Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear variables; close all; clc;

addpath(genpath('../check_utils'))
addpath(genpath('../Part1'))
addpath(genpath('../../ML_toolbox-master'))
addpath(genpath('../../TP2-KMeans-Assignment'))

%% 1a) Load 2d GMM Dataset

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


%% 1b) Load 2d Circle Dataset
clear all; close all; clc;

load('../../TP4-GMM-Datasets/2d-concentric-circles.mat')

% Visualize Dataset
options.class_names = {};
options.title       = '2D Concentric Circles Dataset';
options.labels       = y;

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             2) Check gmm_metrics.m function                %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check gmm_metrics.m function
clc;
check_gmmMetrics(X);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              3) Choosing K test gmm_eval.m                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% K-means Evaluation Parameters
K_range=[1:10]; cov_type = 'iso'; repeats = 10;

% Evaluate gmm-em to find the optimal k
clc;
tic;
gmm_eval(X, K_range, repeats, cov_type);
toc;

%% Pick best K from Plot and Visualize result

% Set GMM Hyper-parameters <== CHANGE VALUES HERE!
K = 4; cov_type = 'diag';  plot_iter = 0;

%%%% Run MY GMM-EM function, estimates the paramaters by maximizing loglik
tic;
[Priors, Mu, Sigma] = my_gmmEM(X, K, cov_type,  plot_iter);
toc;

% Compute GMM Likelihood
[ ll ] = my_gmmLogLik(X, Priors, Mu, Sigma);

% Visualize GMM pdf from learnt parameters
close all;
ml_plot_gmm_pdf(X, Priors, Mu, Sigma)
