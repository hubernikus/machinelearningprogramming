function [ V, L, Mu ] = my_pca( X )
%MY_PCA Step-by-step implementation of Principal Component Analysis
%   In this function, the student should implement the Principal Component 
%   Algorithm following Eq.1, 2 and 3 of Assignment 1.
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o U      : (M x M), Eigenvectors of Covariance Matrix.
%       o L      : (M x M), Eigenvalues of Covariance Matrix
%       o Mu     : (N x 1), Mean Vector of Dataset

% Auxiliary variables
[N, M] = size(X);

% Output variables
V  = zeros(N,N);
L  = zeros(N,N);
Mu = zeros(N);

% ====================== Implement Eq. 1 Here ====================== 
% Centering X around 0
Mu = mean(X,2);  % mean along the second dimension
X = X - repmat(Mu,1,M);

% ====================== Implement Eq.2 Here ======================

% Calculation of the covariance
C = 1/(M-1)*X*X';

% ====================== Implement Eq.3 Here ======================

% Calculation of the eigenvalue-Matrix Lamda and the eigenvector-matrix V
[V,L] = eig(C);

%L2 = zeros(size(L))
%for i = 1:size(C,1)
%    [val,pos] = max(L);
%   V2(:,i) = V(:,pos(2));
%    L2(i,i) = L(pos);
%    L(pos) = 0;
%end
%L = L2;
%V = V2;

% =================== Sort Eigenvectors wrt. EigenValues ==========
% Sort Eigenvalue and get indices
[L_sort, ind] = sort(diag(L),'descend');

% arrange the columns in this order
V=V(:,ind); 

% Vectorize sorted eigenvalues
L = diag(L_sort); 

end

