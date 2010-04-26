function [Inputs,Outputs,ShareOfTrainingSet,Rho,BoolLinKer,Sigmas,PolyParameters] = loadData ()
% This funcion loads the inputs and ouputs from .csv files.
% It also loads the different choices for Kernels from a .xls file.

global NumberOfPoints
%% Inputs and Ouputs
Inputs = csvread('input.txt');
Outputs = csvread('output.txt');
NumberOfPoints = size(Inputs,2);
%% Parameters

Parameters = csvread('parameters.txt',2,1);
SizeOfParameters = size(Parameters,1);

ShareOfTrainingSet = Parameters(1,1); 
Rho = Parameters(2,1);
BoolLinKer = logical(Parameters(3,1));
Sigmas = Parameters(4,:);
PolyParameters = Parameters(5:SizeOfParameters,:);

