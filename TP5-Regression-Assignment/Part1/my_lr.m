function [ w, y_est ] =  my_lr(X, y)
%MY_LR Implementation of the Least-Squares estimation for a linear
%   regression model.
%
%   input -----------------------------------------------------------------
%   
%       o X       : (M x N), a data set with M samples each being of 
%                            dimension N each column corresponds to a datapoint
%       o y       : (M x 1), a vector with outputs y \in R corresponding to X.
%
%   output ----------------------------------------------------------------
%
%       o w       : (N x 1), a vector with estimated linear coefficients of 
%                           of the regressive model y = w'X
%       o y_est   : (M x 1), a vector with estimated outputs y_hat 
%                   corresponding to X.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize variables
[M, N ] = size(X);

% Least square method to calculate w
w =(X'*X)^(-1)*X'*y;

% Estimate y curve with w
y_est = X*w;
end
