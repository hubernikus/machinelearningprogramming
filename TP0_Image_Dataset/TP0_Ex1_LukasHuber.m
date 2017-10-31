%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%           Machine Learning Programming - TP0: Ex 0
%                           Lukas Huber
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

%% 
% Path of dataset folder
datasetPath = './TP0_Image_Dataset';

% seach imagefiles in the folder
imagefiles = dir([datasetPath '/*.jp*g']);

% Number of files
N_files = length(imagefiles);

% Load true labels
load(strcat(datasetPath, '/true_labels.mat'));


% Load images
images = [];
for i = 1:N_files
   images{i,1} = imread([datasetPath '/' imagefiles(i).name]); 
end
    

% Visualize dataset
figure('Color', [1 1 1], 'Name', 'Image Dataset');
cols = ceil(sqrt(N_files));
rows = ceil(sqrt(N_files));

for i = 1:N_files
    subplot(rows,cols,i);
    imagesc(images{i,1});
    axis off;
end

%% 2. Calculate average value of red & blue

mean_red = [];
mean_blue = [];

% Loop over cell array

for i = 1:N_files
    mean_red(i) = mean2(images{i,1}(1:end,1:end,1));
    mean_blue(i) = mean2(images{i}(1:end,1:end,3));  
end

figure('Color', [1 1 1], 'Name', 'True color Features');
scatter(mean_red(true_labels), mean_blue(true_labels),30 ,[1 0 0],'Filled'); 
hold on;
scatter(mean_red(not(true_labels)), mean_blue(not(true_labels)),30 ,[0 0 1],'Filled'); 
hold on;


xlabel('\mu_{red}')
ylabel('\mu_{blue}')
legend('True Red Label', 'True Blue Label')
title('Feature Data with True Labels')
grid on



%% 3. Calucalte true & red belonging

%tic
% true (1) if mu_red > mu_blue
%diff_redBlue =  mean_red - mean_blue;
%y_estimation = logical(abs(diff_redBlue) + diff_redBlue);
%time_calculation = toc

% Evaluation coulor image
y_estimation = logical(zeros(1,N_files));
for i = 1:N_files
    if mean_red(i) > mean_blue(i)
        y_estimation(i) = true;
    end
end

classificationError = 1.0/N_files*sum(abs(y_estimation-true_labels));

scatter(mean_red(y_estimation),mean_blue(y_estimation), 70, [1 0 0]);
hold on;
scatter(mean_red(not(y_estimation)),mean_blue(not(y_estimation)), 70, [0 0 1])
hold on;

plot([0 250],[0 250], 10, [0 0 0])

false_labels = xor(true_labels,y_estimation);

figure('Color', [1 1 1], 'Name', 'Missclassed Images');

rows2 = ceil(sqrt(sum(false_labels)));
cols2 = ceil(sum(false_labels)/rows2);

subplotPos = 0;
for i = 1:N_files
    if(false_labels(i))
        subplotPos = subplotPos + 1
        subplot(cols2,rows2,subplotPos);
        imagesc(images{i,1});
        axis off;
    end;
end;

