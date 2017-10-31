    function [] = plot_eigenvalues( L )
%PLOT_EIGENVALUES Simple plotting function to visualize eigenvalues
%   The student should convert the Eigenvalue matrix to a vector and 
%   visualize the values as a 2D plot.
%   input -----------------------------------------------------------------
%   
%       o L      : (N x N), Diagonal Matrix composed of lambda_i 
%                           
for i = 1:size(L,1)
    l(i) = L(i,i);
end

figure('Name', 'Plot of Eigenvalues', 'Color', [1 1 1])
plot(l, '--', 'Color', [0 0 1])
title('Eigenvalues');
xlabel('Eigenvector Index');
ylabel('Eigenvalues');

end

