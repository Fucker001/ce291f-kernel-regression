% Loads the data set from two CSV format file.
% One for the inputs and one for the outputs.

function [Inputs, Outputs] = loadDataSet()

global NumberOfPoints

%% Read the CSV file.
Inputs = csvread('input.csv')
Outputs = csvread('output.csv')
NumberOfPoints = size(Inputs,2);

%% Check for format compatibility

if (NumberOfPoints ~= size(Outputs,2))
    disp('ERROR, Number of points for the inputs and outputs does not match');
    Inputs = [];
    Outputs = [];
end
    