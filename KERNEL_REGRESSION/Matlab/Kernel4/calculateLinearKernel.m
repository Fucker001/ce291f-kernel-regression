% Calculates and manipulates a Linear Kernel.

function [LinearKernel] = calculateLinearKernel(Inputs)

global NumberOfPoints

%% Calculates the Linear Kernel
Kernel = zeros(NumberOfPoints,NumberOfPoints);

for i = 1:NumberOfPoints
    for j = 1:NumberOfPoints
        Kernel(i,j) = Inputs(:,i)'*Inputs(:,j);
    end
end
csvwrite('linear_kernel.txt',Kernel);


% Normalisation
N = trace(Kernel);
Kernel = Kernel/N;
csvwrite('linear_kernel_trace_normalized.txt',Kernel);

%% Does an SVD and returns the US^0.5 matrix.

[U,S] = svd(Kernel);
LinearKernel = U*(S^0.5);
disp('Linear Kernel SVD finished')

csvwrite('linear_rernel_SVDed.txt',LinearKernel);

toc