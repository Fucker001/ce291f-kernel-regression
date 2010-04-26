function [Error,AggregatedError,Estimate,UStar,Inputs,Outputs,ShareOfTrainingSet] = kernelRegression(share,rho)
% Calculates the estimator and the error
[Inputs,Outputs,ShareOfTrainingSet,Rho,BoolLinKer,...
    Sigmas,PolyParameters]= loadData ();

ShareOfTrainingSet=share;
Rho=rho;

[Kernel] = calculateKernel(Inputs,Rho,...
    BoolLinKer,Sigmas,PolyParameters);

[UStar,PhiStar,VStar] = cvxSolve(Outputs,Kernel,ShareOfTrainingSet);

[Error,AggregatedError,Estimate] = estimator(Outputs,Kernel,PhiStar,VStar,UStar,ShareOfTrainingSet,Rho);