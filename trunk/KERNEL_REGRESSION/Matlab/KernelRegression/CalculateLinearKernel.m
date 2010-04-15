% Calculates and manipulates a Linear Kernel.

function [LinearKernel] = CalculateLinearKernel()

global Inputs NumberOfPoints

%% Calculates the Linear Kernel
Kernel = zeros(NumberOfPoints,NumberOfPoints);

for i = 1:NumberOfPoints
    for j = 1:NumberOfPoints
        Kernel(i,j) = Inputs(:,i)'*Inputs(:,j);
    end
end

% Normalisation
%N = trace(Kernel);
%Kernel = Kernel/N;

%% Does an SVD and returns the US^0.5 matrix.

[U,S] = svd(Kernel);
LinearKernel = U*(S.^0.5);
disp('Linear Kernel SVD finished')
toc