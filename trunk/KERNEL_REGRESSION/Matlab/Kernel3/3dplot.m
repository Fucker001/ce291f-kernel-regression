clear;
clc;

in = csvread('inputs.txt');
out= csvread('outputs.txt');


[X,Y,Z] = peaks(30);
surfc(X,Y,Z)
colormap hsv
axis([-3 3 -3 3 -10 5])


%surf(in(1,:), in(2,:), out(1,:), ones(size(out(1,:))));