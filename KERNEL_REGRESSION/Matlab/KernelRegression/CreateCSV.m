clear;
clc;

n1=100;

O=zeros(1,n1);
I=zeros(1,n1);
for i=1:n1
    I(1,i)=i;
    %I(2,i)=i
    O(i)=i-50;
end

csvwrite('input.csv',I)
csvwrite('output.csv',O)