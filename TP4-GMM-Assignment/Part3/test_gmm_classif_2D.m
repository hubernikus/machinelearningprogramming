%%  Test Implementation of GMM-Classification Algorithm
%    on 2D Datasets.
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Load 2D GMM-Classification Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(genpath('../check_utils'))
addpath(genpath('../Part1'))
addpath(genpath('../../ML_toolbox-master'))
addpath(genpath('../../TP2-KMeans-Assignment'))
addpath(genpath('../../TP3-KNN-Assignment'))


%% (1a) Load 2D data for testing GMM Classification
clear all; close all; clc;

load('../../TP4-GMM-Datasets/2d-concentric-circles.mat')

% Visualize Dataset
options.class_names = {};
options.title       = '2D Concentric Circles Dataset';  
options.labels       = y;

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                   2) Learn GMM for each class              %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% ATTENTION: Load Datatest (1b) before running this code block! %%%%%%
clc; close all;

% Split dataset in train and test for classification
tt_ratio = 0.5;
[ X_train, y_train, X_test, y_test ] = split_data(X, y, tt_ratio );
N_classes = length(unique(y));

% Set GMM Hyper-parameters
K = 5; cov_type = 'full';  plot_iter = 0;

% Learn a GMM for each class
[models] = my_gmm_models(X_train, y_train, K, cov_type, plot_iter);

% Display contours for each class
for c = 1:N_classes
     ml_plot_gmm_pdf(X_train, models(c).Priors, models(c).Mu, models(c).Sigma)
     hold off
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    3) Compute Gaussian Max Likelihood Discriminant Rule    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% ATTENTION: Learn GMM for each class before running this code block! %%%
clc; 

labels = unique(y_test);
% ML discriminant rule
[y_est] = my_gmm_classif(X_test, models, labels, K);

% Compute Accuracy
acc =  my_accuracy(y_test, y_est);

% Plot decision boundary
xplot = linspace(min(X(1,:)), max(X(1,:)), 100)';
yplot = linspace(min(X(2,:)), max(X(2,:)), 100)';

[Xs, Ys] = meshgrid(xplot,yplot);
idx     = my_gmm_classif([Xs(:),Ys(:)]', models, labels, K, [0.5 0.5]);
colors  = hsv(length(unique(idx)));
Z      = reshape(idx,size(Xs));

pcolor(Xs,Ys,Z); shading interp;
colormap(colors);
hold on

% Visualize Split Dataset
options.labels      = y;
options.plot_figure = true;
options.class_names = [];
h2 = ml_plot_data(X',options); hold on;

scatter(X_test(1,y_est == 0),X_test(2,y_est == 0),150,'o','MarkerEdgeColor', [1 0 0]);hold on;
scatter(X_test(1,y_est == 1),X_test(2,y_est == 1),150,'o','MarkerEdgeColor', [0 0 1]);
title (sprintf('My GMM classif, tt-ratio = %1.2f, k = %d, Acc = %1.3f',tt_ratio, K, acc))
hold on

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           4) Creating a dataset of 2 Gaussians             %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 clear;  close all; clc

% Number of datapoints in each class
N_points_class_1 = 5000;
N_points_class_2 = 500;

% Ratio of number of datapoints in each class
cc_ratio = N_points_class_2/N_points_class_1;

% Generate Gaussian data
for i = 1:N_points_class_1
    X_class1(:,i) = normrnd([0 0],[15 15])';
end
for i = 1:N_points_class_2
    X_class2(:,i) = normrnd([80 0],[15 15])';
end
X = [X_class1 X_class2];
y = [ones(1,N_points_class_1) zeros(1,N_points_class_2)];

% Split dataset in train and test for classification
tt_ratio = 0.5;
[ X_train, y_train, X_test, y_test ] = split_data(X, y, tt_ratio );
N_classes = length(unique(y));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      5) Classification without equal class distribution    %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %close all; 
clc

labels = unique(y);
P_class = zeros(1, length(labels));
for label = 1:length(labels)
    P_class(label) = sum(y_train == labels(label))/length(y_train);
end

% Set GMM Hyper-parameters
K = 1; cov_type = 'iso';  plot_iter = 0;

% Learn a GMM for each class
[models] = my_gmm_models(X_train, y_train, K, cov_type, plot_iter);

% We make the assumption of equal variance Sigma in all classes, and equal
% y-coordinate for Mu.
average_Sigma = zeros(size(X,1));
average_Mu = zeros(size(X,1),1);
for c = 1:N_classes
    average_Sigma = average_Sigma + models(c).Sigma;
    average_Mu = average_Mu + models(c).Mu;
end
average_Sigma = average_Sigma./N_classes;
for c = 1:N_classes
     models(c).Sigma = average_Sigma;
     models(c).Mu(2) = average_Mu(2);
end

% ML discriminant rule
[y_est] = my_gmm_classif(X_test, models, labels, K, P_class);

% Compute Accuracy
acc =  my_accuracy(y_test, y_est);

% Plot decision boundary
figure
xplot = linspace(min(X(1,:))-20, max(X(1,:))+20, 1000)';
yplot = linspace(min(X(2,:))-3, max(X(2,:))+3,   1000)';

[Xs, Ys] = meshgrid(xplot,yplot);
idx     = my_gmm_classif([Xs(:),Ys(:)]', models, labels, K, P_class);
colors  = hsv(length(unique(idx)));
Z      = reshape(idx,size(Xs));

pcolor(Xs,Ys,Z); shading interp;
colormap(colors);
hold on

% Visualize Split Dataset
options.labels      = y;
options.plot_figure = true;
options.class_names = [];
h2 = ml_plot_data(X',options);
title(sprintf('My GMM classif, cc-ratio = %1.2f, k = %d, Acc = %1.3f',cc_ratio, K, acc));
print('GMM_classification','-dpng')


%%%%% Without Pclass
figure;
[Xs, Ys] = meshgrid(xplot,yplot);
idx     = my_gmm_classif([Xs(:),Ys(:)]', models, labels, K);%, P_class);
colors  = hsv(length(unique(idx)));
Z      = reshape(idx,size(Xs));

pcolor(Xs,Ys,Z); shading interp;
colormap(colors);
hold on

% Visualize Split Dataset
options.labels      = y;
options.plot_figure = true;
options.class_names = [];
h2 = ml_plot_data(X',options);
title(sprintf('My GMM classif, cc-ratio = %1.2f, k = %d, Acc = %1.3f',cc_ratio, K, acc));
print('GMM_classification_withoutPclass','-dpng')