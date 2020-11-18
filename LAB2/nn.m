%% CLEAR ALL
clear all; close all;

%% GLOBAL VARIABLES
HIDDEN_UNITS=200;
TRAIN_RATIO=0.8;
TEST_RATIO=0.1;
VAL_RATIO=0.1;
EPOCHS=2000;

%% LOAD DATA
load('data/Input.mat');
load('data/Output.mat');

%% SIZE OF DATA20
size(Input)
size(Output)

%% NET INICIALIZATION

net=feedforwardnet([HIDDEN_UNITS]);

%% DATA DIVISION

net.divideFcn = 'dividerand';   % divideFCN allow to change the way the data is
                                % divided into training, validation and test
                                % data sets.
net.divideParam.trainRatio = TRAIN_RATIO; % Ratio of data used as training set
net.divideParam.valRatio = VAL_RATIO;

% Ratio of data used as validation set
net.divideParam.testRatio = TEST_RATIO; % Ratio of data used as test set
net.trainParam.max_fail = 6; % validation check parameter
net.trainParam.epochs=EPOCHS; % number of epochs parameter
net.trainParam.min_grad=1e-5; % minimum performance gradient

%% LAYERS TYPE DEFINITION

net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'softmax';
net.performFcn = 'crossentropy';
net.trainFcn = 'traingdx';

net.trainParam.lr = 0.01;	
net.trainParam.lr_inc = 1.01;	
net.trainParam.lr_dec = 0.7;
net.trainParam.max_perf_inc = 1.03;
net.trainParam.mc = 0.9;
net.inputs{1}.processFcns = {};
net.outputs{end}.processFcns = {};

%% TRAIN

[net,tr,Y,E] = train(net,Input,Output,'useGPU','yes');
performances = [tr.best_perf tr.best_vperf tr.best_tperf]

%% VALIDATION

[~, Yargmax] = max(Y, [], 1);
[~, Oargmax] = max(Output, [], 1);
accVector = Yargmax == Oargmax;

[~, trainMask] = find(tr.trainMask{1} == 1);
[~, valMask] = find(tr.valMask{1} == 1);
[~, testMask] = find(tr.testMask{1} == 1);

trainAcc = sum(accVector(trainMask)) / length(trainMask);
valAcc = sum(accVector(valMask)) / length(valMask);
testAcc = sum(accVector(testMask)) / length(testMask);

validation = [trainAcc, valAcc, testAcc]

%% PLOT

plot(tr.epoch, tr.perf, tr.epoch, tr.vperf, tr.epoch, tr.tperf)
legend("train", "validation", "test")
title("Performance")