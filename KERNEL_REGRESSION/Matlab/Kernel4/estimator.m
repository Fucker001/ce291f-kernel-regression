function [Error,AggregatedError,Estimate] = estimator(Outputs,Kernel,PhiStar,VStar,UStar,ShareOfTrainingSet,Rho)

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
% Krho = MuStar * eye(SizeOfTrainingSet) + KernelStar(1:SizeOfTrainingSet,1:SizeOfTrainingSet);
% AlphaStar = (Krho^(-1)) * Outputs (1,1:SizeOfTrainingSet)';
% Estimate = KernelStar(SizeOfTrainingSet + 1:NumberOfPoints,1:SizeOfTrainingSet) * AlphaStar;

%% Calculates estimate on all the dataset (yes, even on the training)
Krho = MuStar * eye(NumberOfPoints) + KernelStar(1:NumberOfPoints,1:NumberOfPoints);
AlphaStar = (Krho^(-1)) * Outputs (1,1:NumberOfPoints)';
Estimate = KernelStar(1:NumberOfPoints,1:NumberOfPoints) * AlphaStar;

%% Calculates error
SizeOfEstimate = size(Estimate,1);
Error = zeros(1,SizeOfEstimate);
for k = 1:SizeOfEstimate
    Error(1,k) = 100*(abs(Outputs(1,k)-Estimate(k,1)))/Outputs(1,k);
end
AggregatedError=zeros(5,1);
AggregatedError(1)=ShareOfTrainingSet;
AggregatedError(2)=sum(Error)/size(Error,2);
AggregatedError(3)=sum(Error(1,1:SizeOfTrainingSet))/SizeOfTrainingSet;
AggregatedError(4)=sum(Error(1,SizeOfTrainingSet+1:NumberOfPoints))/(NumberOfPoints-SizeOfTrainingSet);
AggregatedError(5)=Rho;
Estimate = Estimate';





