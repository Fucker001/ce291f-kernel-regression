% Calculates the Kernel from different gaussian, linear and polynominial
% kernels.
% The inputs are the desired number of Kernels.
% Out of this number how many are gaussian, linear of polyniminial and each
% array of parameters associated.

function [Kernel] = CalculateKernel(rho,NumberOfKernels,Linear,NumGaussian,GausParam,NumPoly,PolyParam)

global NumberOfPoints

if NumberOfKernels == Linear + NumGaussian + NumPoly 
    
    % Calculates the linear Kernel if it is asked.
    KL = [];
    if islogical(Linear) && Linear
        KL = CalculateLinearKernel;
    else        
        disp('ERROR, incorrect input format for Linear Kernel');
    end
    
    % Calculates and concatenates the Gaussian Kernels if they are asked.
    KG = [];
    if (floor(NumGaussian)-NumGaussian) == 0 && NumGaussian > 0 && size(GausParam,1) == 1
        KG = zeros(NumberOfPoints, NumberOfPoints * NumGaussian);
        for k = 1:NumGaussian
            KG(:,(k-1)*NumberOfPoints + 1:k*NumberOfPoints) = ...
                CalculateGaussianKernel(GausParam(k));
        end
    else
        disp('ERROR, incorrect input format for Gaussian Kernel');
    end

    % Calculates and concatenates the Polynominial Kernels if they are asked.
    KP = [];
    if (floor(NumPoly)-NumPoly) == 0 && NumPoly > 0 && size(PolyParam,2) == 3
        KP = zeros(NumberOfPoints, NumberOfPoints * NumPoly);
        for k = 1:NumPoly
            KP(:,(k-1)*NumberOfPoints + 1:k*NumberOfPoints) = ...
                CalculatePolynominialKernel(PolyParam(k,1), PolyParam(k,2), PolyParam(k,3));
        end
    else
        disp('ERROR, incorrect input format for Polynominial Kernel');
    end

    % Concatenates all the individual Kernels together.
    Kernel = [KL KG KP];
    Kernel = Kernel./rho^2;
else
    disp('ERROR, number of Kernels mismatch');
    Kernel = [];
end
