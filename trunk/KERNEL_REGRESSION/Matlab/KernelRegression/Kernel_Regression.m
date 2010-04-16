% This is the controller for the algorithm.
% It allows to choose the paremeters for each Kernel and the penalty factor
% rho. 
 
function [] = Kernel_Regression()
%% Clears the workspace and the command window.
clear
clc
format compact
tic
%% Declares the data global to be able to pass it on to other functions.

global Inputs Outputs NumberOfPoints

% Loads the data.
[Inputs, Outputs] = loadDataSet();
disp('Data loaded')
toc
%% Initialises the Kernel Parameters.
%rho
rho = 1;

% NumberOfKernels is an integer.
NumberOfKernels = 1;

%Linear is a boolean.
Linear = true;


%NumGaussian is a integer and GausParam is an list of sigmas.
NumGaussian = 1;
GausParam = [0.1];

%NumPoly is an integer and PolyParam is an array of c1, c2, d.
%There is one set of parameters per row.
NumPoly = 0;
PolyParam = [];

%Share for the size of training set
ShareOfTrainingSet=0.95;

%% Calculate the Kernel. Where Number of Kernels
Kernel = CalculateKernel(rho,NumberOfKernels,Linear,NumGaussian,GausParam,NumPoly,PolyParam);

%% Solve the L1 regularized least squares.
SizeOfTrainingSet = ceil(NumberOfPoints*ShareOfTrainingSet);
[UStar, PhiStar, VStar] = Solve(Kernel,SizeOfTrainingSet);

%% Calculate the estimation.
[Estimate] = CalculateEstimate(Kernel,SizeOfTrainingSet,UStar,PhiStar,VStar);

%% Calculate error.
Error = CalculateError(Estimate);
Error = sum(Error,1)/size(Error,1);

%% Display results
disp('**************************************************************')
disp('Your Estimate is :')
disp(Estimate')
disp(' ')
disp('Your Error in percent is :')
disp(Error')
toc

%% plot

plot(Inputs(1,SizeOfTrainingSet+1:NumberOfPoints),Outputs(1,SizeOfTrainingSet+1:NumberOfPoints),Inputs(1,SizeOfTrainingSet+1:NumberOfPoints),Estimate)
