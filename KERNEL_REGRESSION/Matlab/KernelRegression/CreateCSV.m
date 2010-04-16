clear;
clc;

n1=100;

O=zeros(1,n1);
I=zeros(2,n1);
for i=1:n1
    I(1,i)=i;
    I(2,i)=1;
    O(i)=5+i/20+2*cos(2*i*pi/100);
end

csvwrite('input.csv',I)
csvwrite('output.csv',O)