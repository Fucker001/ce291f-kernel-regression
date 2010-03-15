% This is the controller for the algorithm.
% It allows to choose the paremeters for each Kernel and the penalty factor
% rho. 
 
function [] = Kernel_Regression()
%% Clears the workspace and the command window.
clear
clc

%% Declares the data global to be able to pass it on to other functions.

global Inputs Outputs NumberOfPoints

% Loads the data.
[Inputs, Outputs] = loadDataSet();

%% Initialises the Kernel Parameters.
%rho
rho = 0.5;

% NumberOfKernels is an integer.
NumberOfKernels = 4;

%Linear is a boolean.
Linear = true;

%NumGaussian is a integer and GausParam is an list of sigmas.
NumGaussian = 3;
GausParam = [1 10 100];

%NumPoly is an integer and PolyParam is an array of c1, c2, d.
%There is one set of parameters per row.
NumPoly = 0;
PolyParam = [];

%% Calculate the Kernel. Where Number of Kernels
Kernel = CalculateKernel(rho,NumberOfKernels,Linear,NumGaussian,GausParam,NumPoly,PolyParam);

%% Solve the L1 regularized least squares.
SizeOfTrainingSet = ceil(NumberOfPoints/4);
[UStar, PhiStar, VStar] = Solve(Kernel,SizeOfTrainingSet);

%% Calculate the estimation.
[Estimate] = CalculateEstimate(Kernel,SizeOfTrainingSet,UStar,PhiStar,VStar);

%% Calculate error.
Error = CalculateError(Estimate);

%% Display results

disp('**************************************************************')
disp('Your Estimate is :')
disp(Estimate')
disp(' ')
disp('Your Error in percent is :')
disp(Error')