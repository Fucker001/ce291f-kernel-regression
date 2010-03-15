% Calculates the error element wize.

function [Error] = CalculateError (Estimate)

global Outputs NumberOfPoints

%% Initializes the error matrix.
SizeOfTestSet = size(Estimate,1);
Error = zeros(SizeOfTestSet,1);

%% Sets the error matrix.

for k = 1:SizeOfTestSet
    Error(k,1) = 100*(abs(Outputs(1,NumberOfPoints - SizeOfTestSet +k)...
        - Estimate(k,1)))/Outputs(1,NumberOfPoints - SizeOfTestSet +k);
end