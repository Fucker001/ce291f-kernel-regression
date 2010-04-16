% Calculates the estimate using many steps.

function [Estimate] = CalculateEstimate(Kernel,SizeOfTrainingSet,UStar,PhiStar,VStar)

global NumberOfPoints Outputs

%% Sets coefficient.
MuStar = VStar / PhiStar;
b = Outputs (1,1:SizeOfTrainingSet)';
disp('MuStar is equal to')
disp(MuStar);
%% Sets the coefficients for the linear combinaison of eigenvectors of the general Kernel.
N = size(UStar,1);
LambdaStar = zeros (1,N);
for k = 1:N
    LambdaStar(k,1) = abs(UStar(k,1)) / PhiStar;
end

%% Sets the optimal Kernel.
KernelStar = zeros (NumberOfPoints);
for k = 1:N
    KernelStar = KernelStar + LambdaStar(k,1)*(Kernel(:,k)*Kernel(:,k)');
end
disp('Kernel* done.')
toc
%% Calculates estimate.
Krho = MuStar * eye(SizeOfTrainingSet) + KernelStar(1:SizeOfTrainingSet,1:SizeOfTrainingSet)/rho^2;
AlphaStar = (Krho^(-1)) * b;
Estimate = KernelStar(SizeOfTrainingSet + 1:NumberOfPoints,1:SizeOfTrainingSet) * AlphaStar;
