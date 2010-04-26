function [Error,Estimate,UStar,Inputs,Outputs,ShareOfTrainingSet] = kernelRegression()
% Calculates the estimator and the error
[Inputs,Outputs,ShareOfTrainingSet,Rho,BoolLinKer,...
    Sigmas,PolyParameters]= loadData ();

[Kernel] = calculateKernel(Inputs,Rho,...
    BoolLinKer,Sigmas,PolyParameters);

[UStar,PhiStar,VStar] = cvxSolve(Outputs,Kernel,ShareOfTrainingSet);

[Error,Estimate] = estimator(Outputs,Kernel,PhiStar,VStar,UStar,ShareOfTrainingSet);