function [Kernel] = calculateKernel(Inputs,Rho,BoolLinKer,Sigmas,PolyParameters)

NumberOfPoints = size(Inputs,2);

%% Calculates the linear Kernel if it is asked.
KL = [];
if BoolLinKer
    KL = calculateLinearKernel(Inputs);
end

%% Calculates and concatenates the Gaussian Kernels if they are asked.
KG = [];
if sum(Sigmas,2) ~= 0
    KG = zeros(NumberOfPoints, NumberOfPoints * nnz(Sigmas));
    for k = 1:nnz(Sigmas)
        KG(:,(k-1)*NumberOfPoints + 1:k*NumberOfPoints) = ...
            calculateGaussianKernel(Inputs,Sigmas(k));
    end
end

%% Calculates and concatenates the Polynomial Kernels if they are asked.
KP = [];
if sum(sum(PolyParameters))~=0
    KP = zeros(NumberOfPoints, NumberOfPoints*nnz(PolyParameters)/3);
    for k = 1:nnz(PolyParameters)/3
        KP(:,(k-1)*NumberOfPoints + 1:k*NumberOfPoints) = ...
            calculatePolynomialKernel(Inputs,PolyParameters(k,1), PolyParameters(k,2), PolyParameters(k,3));
    end
end

%% Concatenates all the individual Kernels together.
Kernel = [KL KG KP];
Kernel = Kernel/Rho^2;

csvwrite('fat_K.txt',Kernel);