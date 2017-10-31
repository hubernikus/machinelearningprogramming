function [ y_est ] =  my_knn(X_train,  y_train, X_test, k, type)
%MY_KNN Implementation of the k-nearest neighbor algorithm
%   for classification.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o k        : number of 'k' nearest neighbors
%       o type   : (string), type of distance {'L1','L2','LInf'}
%
%   output ----------------------------------------------------------------
%
%       o y_est   : (1 x M_test), a vector with estimated labels y \in {0,1} 
%                   corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Matrix dimensions
[N, M_train] = size(X_train);
[~, M_test] = size(X_test);

% Extract number 
class_found = unique(y_train); 

% Distance Matrix [M_test x M_train] with distance from each testing 
% datapoint to training data point
dist_X = zeros(M_test, M_train);
for i = 1:M_test
    for j = 1:M_train
        dist_X(i,j) = my_distance(X_train(:,j),X_test(:,i),type);
    end
end

% Sort the Matrix for each M_test
% index_distMax [M_test x M_train] contains the index of the correspondin 
% training matrix
[~, index_distMax] = sort(dist_X,2); 

% Count the number of points that belong to one class
test_classCount= zeros(M_test, length(class_found));
k = min(k, M_train); %limit k to not exeed index of index_distMax
for i = 1:length(class_found)
    if k ==  1 % If k==1 y_train converts to row vector
        test_classCount(:,i) = (sum(y_train(index_distMax(:,1:k))' == class_found(i),2));
    else
        test_classCount(:,i) = (sum(y_train(index_distMax(:,1:k)) == class_found(i),2));
    end
end

% Calculate the index of the class with the maximum points assigned to X_test
% [M_test x n_class]
[~,max_class] = max(test_classCount,[],2);

% Get the corresponding class index
y_est = (class_found(max_class));

end