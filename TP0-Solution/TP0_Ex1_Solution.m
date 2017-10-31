%%  Solution to Exercise 1 of TP0 (Matlab Programming Basics)
clear all;
close all;
clc;

%% 1) Load Image Dataset and Visualize images
%% 1st way of loading images (using dir function)
% Path of Dataset Folder
dataset_path    = './TP0_Image_Dataset';

% Search for file names within folder
imagefiles = dir([dataset_path '/*.jp*g']);

% Number of files found
M = length(imagefiles);    

% Load True labels
load(strcat(dataset_path,'/true_labels.mat'))

% Load images as a structure of 3D matrices
images = [];
for i = 1:M    
    filename = imagefiles(i).name;
    image = imread([dataset_path '/' filename]);    
    images{i,1} = image;
end

% Visualize image dataset
figure('Color', [1 1 1], 'Name', 'Image Dataset')
cols     = ceil(sqrt(M));
rows     = floor(sqrt(M));

for ii=1:M        
    subplot(rows,cols,ii)        
    imagesc(images{ii,1})
    axis off    
end

%% 2nd way of loading images (loading images by name)
M = 56;
for i = 1:M    
    image = imread(num2str(i),'jpg');    
    images{i,1} = image;    
end

% Load True labels
load('/true_labels_2.mat')

% Visualize image dataset
figure('Color', [1 1 1], 'Name', 'Image Dataset')
cols     = ceil(sqrt(M));
rows     = floor(sqrt(M));
use_axis = false;

for ii=1:M        
    subplot(rows,cols,ii)        
    imagesc(images{ii,1})
    axis off    
end

%% 2) Extract Features from Image Dataset

mu_r = [];
mu_b = [];

% Loop over cell array of images to compute mu_R and mu_B
for i = 1:M

    % Compute mu_r and mu_b for current image
    mu_r_ =  mean2(images{i}(1:end,1:end,1));
    mu_b_ =  mean2(images{i}(1:end,1:end,3));
    
    % Load Vectors mu_R and mu_B
    mu_r = [mu_r mu_r_];
    mu_b = [mu_b mu_b_];

end

% Plot Features
figure('Color',[1 1 1], 'Name', 'Mean Color Features')
scatter(mu_r(true_labels), mu_b(true_labels), 40, [1 0 0], 'Filled'); hold on
scatter(mu_r(true_labels==0), mu_b(true_labels==0), 40, [0 0 1], 'Filled'); hold on
xlabel('\mu_r')
ylabel('\mu_b')
grid on
legend('True Red Label', 'True Blue Label')
title('Feature Data with True Labels')

%% 3) Classification based on Mean Color Features

%%%%% Apply Decision Function, estimate labels %%%%%

% Classification rule
y_est = mu_r > mu_b;

% Plot estimated labels on top of feature data
figure('Color',[1 1 1], 'Name', 'Estimated Labels on Feature Data')

% Plot Feature Data with True labels
scatter(mu_r(true_labels), mu_b(true_labels), 40, [1 0 0], 'Filled'); hold on
scatter(mu_r(true_labels==0), mu_b(true_labels==0), 40, [0 0 1], 'Filled'); hold on

% Plot Estimated Labels on top of Feature Data
scatter(mu_r(y_est), mu_b(y_est), 120, [1 0 0]); hold on
scatter(mu_r(y_est==0), mu_b(y_est==0), 120, [0 0 1]); hold on

% Plot Decision Boundary
plot (0:255,0:255, 'k');
xlabel('\mu_r')
ylabel('\mu_b')
axis tight
grid on
legend('True Red Label', 'True Blue Label', 'Estimated Red Label', 'Estimated Blue Label', 'Decision Boundary')
title('Estimated Labels on Feature Data')

%%%%% Estimate Classification Error %%%%%
N_errors = sum(abs(true_labels - y_est));
class_error =  N_errors / M;
clc
fprintf('Classfication Error: %f \n', class_error);

% Visualize misclassified images
misclass = true_labels ~= y_est;
figure('Color', [1 1 1], 'Name', 'Misclassified Images')
cols     = ceil(sqrt(N_errors));
rows     = floor(sqrt(N_errors));
iter = 1;
for ii=1:M
    if misclass(ii)
        misclass_image = images{ii};
        subplot(rows,cols,iter)        
        imagesc(images{ii,1})
        axis off           
        iter = iter + 1;
    end
end
