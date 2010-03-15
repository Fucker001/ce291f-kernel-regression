% Calculates and manipulates a gaussian Kernel of parameter sigma.
% Does an SVD and returns the US^0.5 matrix.

function [GaussianKernel] = CalculateGaussianKernel(sigma)

global Inputs NumberOfPoints

%% Calculates the Gaussian Kernel
Kernel = zeros(NumberOfPoints,NumberOfPoints);

for i = 1:NumberOfPoints
    for j = 1:NumberOfPoints
        Kernel(i,j) = (norm(Inputs(:,i)-Inputs(:,j),2))^2;
    end
end
Kernel = exp(-Kernel/(2*sigma^2));

% Normalisation
N = trace(Kernel);
Kernel = Kernel/N;

%% Does an SVD and returns the US^0.5 matrix.

[U,S] = svd(Kernel);
GaussianKernel = U*(S.^0.5);
