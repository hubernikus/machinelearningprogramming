%%  Solution to Exercise 2 of TP0 (Matlab Programming Basics)
clear all;
close all;
clc;

%% Solve System of linear equations Ax = b
%  by x = A/b, where our system is the following:
%   x +  y - z  = 1
%  2x +  y + z  = 1
%  -x - 2y + 5z = 2

% Load A with equation paramaters
A = [1 1 -1; 2 1 1; -1 -2 5]
b = [1; 1; 2]
x = A\b