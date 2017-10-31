function [ p ] = explained_variance( L, Var )
%EXPLAINED_VARIANCE Function that returns the optimal p given a desired
%   explained variance. The student should convert the Eigenvalue matrix 
%   to a vector and visualize the values as a 2D plot.
%   input -----------------------------------------------------------------
%   
%       o L      : (N x N), Diagonal Matrix composed of lambda_i 
%  
%   output ----------------------------------------------------------------
%
%       o p      : optimal principal components wrt. explained variance


% ====================== Implement Eq. 8 Here ====================== [
N = size(L,1);
expl_var        = ones(1,N)*L./sum(sum(L));

% ====================== Implement Eq. 9 Here ====================== 
%cum_expl_var    = expl_var + [0 expl_var(1:N-1)];
cum_expl_var = expl_var;
for i = 2:N
    cum_expl_var(i) = cum_expl_var(i-1) + cum_expl_var(i);
end

% ====================== Implement Eq. 10 Here ====================== 
p = N;  
for i = 1:N
    if cum_expl_var(i) > Var
        p = i;
        break;
    end
end

% Visualize Explained Variance from Eigenvalues
figure;
plot(cum_expl_var, '--r', 'LineWidth', 2) ; hold on;
plot(p,cum_expl_var(p),'or')
title('Explained Variance from EigenValues')
ylabel(' Cumulative Variance Explained')
xlabel('Eigenvector index')
grid on

end

