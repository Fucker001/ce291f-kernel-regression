function [Error,Estimate] = estimator(Outputs,Kernel,PhiStar,VStar,UStar,ShareOfTrainingSet)

global NumberOfPoints 

%% Sets coefficient.
MuStar = VStar / PhiStar;
SizeOfTrainingSet = floor(ShareOfTrainingSet*NumberOfPoints);
%% Sets the coefficients for the linear combinaison of eigenvectors of the general Kernel.
N = size(UStar,1);
LambdaStar = zeros (N,1);
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
Krho = MuStar * eye(SizeOfTrainingSet) + KernelStar(1:SizeOfTrainingSet,1:SizeOfTrainingSet);
AlphaStar = (Krho^(-1)) * Outputs (1,1:SizeOfTrainingSet)';
Estimate = KernelStar(SizeOfTrainingSet + 1:NumberOfPoints,1:SizeOfTrainingSet) * AlphaStar;

%% Calculates error
SizeOfTestSet = size(Estimate,1);
Error = zeros(1,SizeOfTestSet);
for k = 1:SizeOfTestSet
    Error(1,k) = 100*(abs(Outputs(1,NumberOfPoints - SizeOfTestSet +k)...
        - Estimate(k,1)))/Outputs(1,NumberOfPoints - SizeOfTestSet +k);
end

Estimate = Estimate';