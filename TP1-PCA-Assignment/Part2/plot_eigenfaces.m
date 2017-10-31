function [V, L, Mu] = plot_eigenfaces(X, sizeIm)
%PLOT_EIGENFACES Extracts and displays eigenfaces based on dataset X
%   
%   
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%
%   output ----------------------------------------------------------------
%
%       o A_p      : (p x N), Projection Matrix.
%       o Y      : (p x M), Projected data set with N samples each being of dimension k.


% Extract Principal Directions and Components
[V, L, Mu] = my_pca(X);


% Display the first 20 Eigenfaces
N_displayed_images = 20;


figure('Name','First 20 Eigenfaces','Color',[1 1 1], 'pos', [200 400 1000 200]);
%cols = ceil(sqrt(N_displayed_images));
cols = 10;
rows = ceil(N_displayed_images/cols);
colormap('Gray');

for i = 1:N_displayed_images

        % Extract Eigenface
        eigenface = reshape(V(:,i), sizeIm, sizeIm);
        % Plot Eigenface
        subplot(rows,cols,i);
        imagesc(eigenface);
        axis off;
        
end

end