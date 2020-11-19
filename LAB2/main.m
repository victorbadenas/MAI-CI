%% CLEAN
clear all; close all;

%% LOAD CONFIG JSON
fname = 'configuration/configs.json';
configs = jsondecode(fileread(fname));

%% NUMBER OF PARAMETERS TO LOOP INTO
numFunctions = length(configs.functions);
numHiddenUnits = length(configs.hiddenUnits);
numPercentages = length(configs.percentages);

%% LOAD DATA
load('data/Input.mat');
load('data/Output.mat');

%% CONSTANT VALUES
EPOCHS = 1000;
NRUNS = 5;
GPU = 1;
trainFcn = 'traingdm';
trainFcnParams = struct('lr',0.1,'mc',0.8);

%% INIT ACCURACIES AND PERFORMANCES
allPerformances = zeros(3,numPercentages,numHiddenUnits,numFunctions);
allAccuracies = zeros(3,numPercentages,numHiddenUnits,numFunctions);

%% LOOP FOR ALL FUNCTION CONFIGS, HIDDEN UNITS AND PERCENTAGE SPLITS
for pIdx = 1:numPercentages
    percentages = configs.percentages(pIdx);
    for hIdx = 1:numHiddenUnits
        hiddenUnits = configs.hiddenUnits(hIdx);
        for fIdx = 1:numFunctions
            fprintf("Currently training with %i percentages, %i hiddenUnits, %i numFunctions\n", pIdx, hIdx, fIdx)
            functions = configs.functions(fIdx);
            [performances, accuracies]=trainNN(Input,Output,EPOCHS,functions,hiddenUnits,percentages,NRUNS,trainFcn,trainFcnParams,GPU)
            allPerformances(:,pIdx,hIdx,fIdx) = performances;
            allAccuracies(:,pIdx,hIdx,fIdx) = accuracies;
        end
    end
end

%% SAVE AS CSV AND MAT
save('metrics_.mat','allPerformances','allAccuracies');
mkdir csv_acc_
mkdir csv_perf_
save4Dcsv(allAccuracies,'csv_acc_',["Train" "Val" "Test"],["80-10-10" "40-20-40" "10-10-80"])
save4Dcsv(allPerformances,'csv_perf_',["Train" "Val" "Test"],["80-10-10" "40-20-40" "10-10-80"])