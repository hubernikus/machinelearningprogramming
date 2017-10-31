    function [ X_train, y_train, X_test, y_test ] = split_data(X, y, tt_ratio )
%SPLIT_DATA Randomly partitions a dataset into train/test sets using
%   according to the given tt_ratio
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y        : (1 x M), a vector with labels y \in {0,1} corresponding to X.
%       o tt_ratio : train/test ratio.
%   output ----------------------------------------------------------------
%
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dimension variables of the system
[~, M] = size(X);
M_train = floor(tt_ratio*M); % Number of data points for training

% Get random indexes of the dataset for the test and train elements 
index_M = randperm(M);
index_train = index_M(1:M_train);
index_test = index_M(M_train+1:end);


% Extracting Data
X_train = X(:,index_train);
y_train = y(index_train);

X_test = X(:,index_test);
y_test = y(index_test);

end

