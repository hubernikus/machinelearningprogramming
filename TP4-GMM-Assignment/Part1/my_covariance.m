function [ Sigma ] = my_covariance( X, Mu, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o Mu    : (N x 1), an Nx1 matrix corresponding to the centroid mu_k \in R^Nn
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%

% Assign Paramters
[N,M] = size(X);

% Check if Dimension N = 2
if not(N==2)
    warning('Dimension is not equal to two. This does exeed model!');
end
if M==1
    warning('Dimension M=1 can lead to singularities in caclulation');
end

% Center Data around mean value
X = X - ndgrid(Mu,zeros(1,M));
%Sigma = zeros(2);

switch type
    case 'full' % Full covariance Matrix
        Sigma = 1./(M-1)*(X*X');
        return;
    case 'diag' % Diagonal covariance Matrix
        Sigma = 1./(M-1)*(X*X') .* diag(ones(1,N)); % Delete non diagonal elements
        return;
    case 'iso'
        Sigma = 1./(N*M)*sum(sum(X.^2))*diag(ones(1,N));
        return;
end

error(['Error: Type <<', type, '>> does not exist!'])

end

