%% CLEAR ALL

clear all; close all;

%% GLOBAL VARIABLES

HIDDEN_UNITS=200;
TRAIN_RATIO=0.8;
TEST_RATIO=0.1;
VAL_RATIO=0.1;
EPOCHS=1000;

%% LOAD DATA

load('data/Input.mat');
load('data/Output.mat');

%% SIZE OF DATA

size(Input);
size(Output);

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

%% LAYERS TYPE DEFINITION

net.trainFcn = 'traincgb';

net.trainParam.max_fail = 6; % validation check parameter
net.trainParam.epochs = EPOCHS; % number of epochs parameter
net.trainParam.min_grad = 1e-5; % minimum performance gradient

net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'softmax';
net.performFcn = 'crossentropy';

net.inputs{1}.processFcns = {};
net.outputs{end}.processFcns = {};

%% TRAIN

[net,tr,Y,E] = train(net,Input,Output,'useGPU','yes');
performances = [tr.best_perf tr.best_vperf tr.best_tperf];

%% VALIDATION

[~, Yargmax] = max(Y, [], 1);
[~, Oargmax] = max(Output, [], 1);
accVector = Yargmax == Oargmax;

trainAcc = sum(accVector(tr.trainInd)) / length(tr.trainInd);
valAcc = sum(accVector(tr.valInd)) / length(tr.valInd);
testAcc = sum(accVector(tr.testInd)) / length(tr.testInd);

accuracies = [trainAcc, valAcc, testAcc]

%% PLOT

figure()
plot(tr.epoch, tr.perf, tr.epoch, tr.vperf, tr.epoch, tr.tperf)
legend("train", "validation", "test")
title("Performance")
grid on
xlabel epochs
ylabel performance

%% CONFUSION MATRIX

figure()
confmat = confusionmat(Yargmax(tr.testInd), Oargmax(tr.testInd));
imshow(confmat./max(confmat))
csvwrite("confusion_matrix.csv", confmat);
