%% CLEAR ALL
clear all; close all;

%% GLOBAL VARIABLES
HIDDEN_UNITS=50;
TRAIN_RATIO=0.1;
TEST_RATIO=0.1;
VAL_RATIO=0.8;
EPOCHS=1;

%% LOAD DATA
load('data/Input.mat');
load('data/Output.mat');
% load('data/caltech101_silhouettes_16.mat');
% load('data/caltech101_silhouettes_16_split1.mat');
% load('data/caltech101_silhouettes_28.mat');
% load('data/caltech101_silhouettes_28_split1.mat');
% Input=Input(:, 1:200);
% Output=Output(:, 1:200);

%% SIZE OF DATA20
size(Input)
size(Output)

%% NET INICIALIZATION

% net=perceptron;
net=feedforwardnet([20, 2]);
view(net)
%% DATA DIVISION

net.divideFcn = 'dividerand';   % divideFCN allow to change the way the data is
                                % divided into training, validation and test
                                % data sets.
net.divideParam.trainRatio = TRAIN_RATIO; % Ratio of data used as training set
net.divideParam.valRatio = VAL_RATIO;

% Ratio of data used as validation set
net.divideParam.testRatio = TEST_RATIO; % Ratio of data used as test set
% net.trainParam.max_fail = 6; % validation check parameter
net.trainParam.epochs=EPOCHS; % number of epochs parameter
% net.trainParam.min_grad=1e-5; % minimum performance gradient

%% LAYERS TYPE DEFINITION

% net.layers{1}.transferFcn = 'logsig';
% net.layers{1}.transferFcn = 'tansig';
% net.layers{2}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'softmax';
% net.performFcn = 'mse';
% net.performFcn = 'crossentropy';

% net.trainFcn = 'trainlm';    % Levenberg-Marquardt
% net.trainFcn = 'traingdm';   % Gradient Descent with momentum
% net.trainFcn = 'traingdx';   % Gradient descent with momentum and adaptive
%                             % learning rate backpropagation

% net.trainParam.mc = 0.8; % momentum parameter
% net.trainParam.lr = 0.01; % learning rate parameter
net.inputs{1}.processFcns = {'mapminmax'};

%% TRAIN

% [net,tr,a,b] = train(net,Input,Output);
[net,tr,Y,E] = train(net,Input,Output,'useGPU','yes');

% net = configure(net,Input,Output);
% Xgpu = gpuArray(Input);
% Tgpu = gpuArray(Output);
% net = train(net,Xgpu,Tgpu);
% Ygpu = net(Xgpu);
% Y = gather(Ygpu); 
view(net)
