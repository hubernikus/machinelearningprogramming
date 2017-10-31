%%  Test Implementation of Principal Component Analysis (PCA)
%    on 2D Datasets. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    Load YALE Face Dataset                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;

% Dataset Path
dataset_path = '../../TP1-PCA-Dataset/Faces/';

% Load 2D Testing Dataset for PCA
load(strcat(dataset_path,'Yale_32x32.mat'))

% Generate Variables
X       = fea;
labels  = gnd'; 
[M, N]  = size(X);
sizeIm  = sqrt(N);

% Plot 64 random samples of the dataset
idx = randperm(size(X,1));
h0  = ml_plot_images(X(idx(1:64),:),[sizeIm sizeIm]);

% Transpose for PCA
X = fea';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             Test your plot_eigenfaces.m function            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function plots the first 20 eigenfaces and returns the 
% projections variabls from my_pca

[V, L, Mu] = plot_eigenfaces(X, sizeIm);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Test your reconstruction_eigenfaces.m function     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function reconstructs the first face image from the dataset with
% p = 1, p = 51, p = 101, p = 151, it should display these images +
% the mean image of the dataseta and the original first image

reconstruction_eigenfaces(X, V, Mu, sizeIm)
