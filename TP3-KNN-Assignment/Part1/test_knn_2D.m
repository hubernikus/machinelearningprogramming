%%  Test Implementation of K-NN (K-Nearest Neighbors)
%    on 2D Datasets. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           1) Load 2D KNN Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Load Concentric Circle Data
clear all;
close all;
clc;

load('../../TP3-KNN-Datasets/2d-concentric-circles.mat')

% Visualize Dataset
options.labels      = y;
options.class_names = {'y = 0','y = 1'};
options.title       = '2D Concentric Circles Dataset';

h0 = ml_plot_data(X',options);
axis equal

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     2) Data Handling for Classification (split_data.m)        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Training/Testing Ratio
tt_ratio = 0.3; % 30% of points = train / 70% of points = testing

% Split data into a training dataset that kNN can use to make predictions 
% and a test dataset that we can use to evaluate the accuracy of the model.
[X_train, y_train, X_test, y_test] = split_data(X, y, tt_ratio);

% Check data_split.m function
clc;
check_datasplits(X, tt_ratio, X_train, X_test);

% Visualize Split Dataset
options.labels      = y_train;
options.class_names = [];
options.title       = sprintf('Data Split for 2D Dataset tt-ratio: %1.2f',tt_ratio);

h1 = ml_plot_data(X_train',options); hold on;
scatter(X_test(1,:),X_test(2,:),50,'^','MarkerFaceColor',[1 1 0.5],'MarkerEdgeColor', [0 0 0]);
legend({'$y=0$','$y = 1$','$\mathbf{x}\prime$'},'Interpreter','latex')
axis equal

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                  3) Test my_knn.m function                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check my_knn.m function
clc;
check_myknn(X_train, y_train, X_test);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             4) Test my_accuracy.m function                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select k
k = 5; 

% Compute y_estimate from k-NN
y_est =  my_knn(X_train, y_train, X_test, k, 'L2');

% Check accuracy.m function
clc;
check_myaccuracy(y_test, y_est);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           5) Visualize kNN Results and Accuracy            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;

% Select k
k = 35; d_type = 'L2';

% Compute y_estimate from k-NN
y_est =  my_knn(X_train, y_train, X_test, k, d_type);

% Compute Accuracy
acc =  my_accuracy(y_test, y_est);

% Visualize Split Dataset
options.labels      = y;
options.class_names = [];
options.title       = sprintf('My kNN, tt-ratio = %1.2f, k= %d, Acc = %1.3f',tt_ratio, k, acc);

h2 = ml_plot_data(X',options); hold on;
scatter(X_test(1,y_est == 0),X_test(2,y_est == 0),150,'o','MarkerEdgeColor', [1 0 0]);hold on;
scatter(X_test(1,y_est == 1),X_test(2,y_est == 1),150,'o','MarkerEdgeColor', [0 0 1]);
legend({'$y=0$','$y = 1$','$\hat{y} = 0$','$\hat{y} = 1$'},'Interpreter','latex')
axis equal

%% Plot K-NN Decision boundary
knn_options.k      = k;
knn_options.d_type = d_type;
[~, model]= knn_classifier(X_train, y_train, [], knn_options);
f_knn     = @(X_test)knn_classifier(X_test, [], model, []);

% Plot Decision Boundary
clc;
c_options         = [];
plot_data_options = [];

c_options.dim_swaped     = true;
c_options.show_misclass  = false;
c_options.title          = sprintf('K(%d)-NN Decision Boundary with TT/ratio: %1.2f',k,tt_ratio);

if exist('hc','var') && isvalid(hc), delete(hc);end
hc = ml_plot_classifier(f_knn,X',y,c_options,plot_data_options);
axis tight


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         6) Choosing K by visualizing knn_eval.m            %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Select Range of K to test accuracy
M_train = length(X_train);
k_range = [1:2:ceil(M_train/2)];

knn_eval(X_train, y_train, X_test, y_test, k_range); 