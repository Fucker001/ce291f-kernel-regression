% Calculates and manipulates a gaussian Kernel of parameter sigma.
% Does an SVD and returns the US^0.5 matrix.

function [GaussianKernel] = calculateGaussianKernel(Inputs,Sigma)

global NumberOfPoints

%% Calculates the Gaussian Kernel
Kernel = zeros(NumberOfPoints,NumberOfPoints);

for i = 1:NumberOfPoints
    for j = 1:NumberOfPoints
        Kernel(i,j) = (norm(Inputs(:,i)-Inputs(:,j),2))^2;
    end
end
Kernel = exp(-Kernel/(2*Sigma^2));
csvwrite('gaussian_kernel.txt',Kernel);

% Normalisation
N = trace(Kernel);
Kernel = Kernel/N;
csvwrite('gaussian_kernel_trace_normalized.txt',Kernel);

%% Does an SVD and returns the US^0.5 matrix.

[U,S] = svd(Kernel);
csvwrite('gaussian_kernel_U.txt',U);
csvwrite('gaussian_kernel_S.txt',S);
csvwrite('gaussian_kernel_S_squareroot.txt',S^0.5);
GaussianKernel = U*(S^0.5);
disp('Gaussian Kernel SVD finished')

csvwrite('gaussian_kernel_SVDed.txt',GaussianKernel);

toc