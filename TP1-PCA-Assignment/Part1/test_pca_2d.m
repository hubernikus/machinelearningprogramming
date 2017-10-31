%%  Test Implementation of Principal Component Analysis (PCA)
%    on 2D Datasets. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               Load 2D PCA Testing Dataset                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;

%path_to_ML_toolbox = C:\Users\lukas\Google Drive\EPFL\MachineLearningProgramming\Ex1\ML_toolbox-master;
%addpath(genpath('path_to_ML_toolbox'));
% addpath(genpath('../../ML_toolbox-master'))

% Dataset Path
dataset_path = '../../TP1-PCA-Dataset/';

% Load 2D Testing Dataset for PCA
load(strcat(dataset_path,'2D_Gaussian.mat'))

% Visualize Dataset
options.labels      = labels;
options.title       = 'X = 2D Random Gaussian';

h0 = ml_plot_data(X',options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                   Test my_pca.m function                   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract Principal Directions and Components
[V, L, Mu] = my_pca(X);

% Test my_pca.m against ML_toolbox numerically
test_mypca(X, V, L, Mu)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                     Test project_pca.m                     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Project Data to Choosen Principal Components
p = 2;
[A_p, Y] = project_pca(X, Mu, V, p );

% Visualize Projected Data
plot_options             = [];
plot_options.is_eig      = true;
plot_options.labels      = labels;
plot_options.class_names = '2D Gauss';
plot_options.title       = 'My Projected data PCA';

h1 = ml_plot_data(Y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               Test reconstruct_pca.m                   %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Project Data to 1d
p = 1;
[A_p, Y] = project_pca(X, Mu, V, p); 

% Reconstruct Lossy Data from 1d
[X_hat]  = reconstruct_pca(Y, A_p, Mu); % ?!?

% Estimate Reconstruction Error
[e_rec]  = reconstruction_error(X, X_hat);
fprintf('Reconstruction Error with p=%d is %f \n',p,e_rec);

% Visualize Reconstructed Data
options.labels      = labels;
options.title       = 'Xhat : Reconstructed Data';
h0 = ml_plot_data(X_hat',options);
axis equal;
