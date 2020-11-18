clear all; close all;

fname = 'configuration/configs.json';
configs = jsondecode(fileread(fname));

numFunctions = length(configs.functions);
numHiddenUnits = length(configs.hiddenUnits);
numPercentages = length(configs.percentages);

load('data/Input.mat');
load('data/Output.mat');

EPOCHS = 1000;
NRUNS = 3;
GPU = 1;
trainFcn = 'traingdx';
trainFcnParams = struct('lr',0.01, 'lr_inc', 1.01, 'lr_dec', 0.7, 'max_perf_inc', 1.03, 'mc', 0.9);

allPerformances = zeros(3,numPercentages,numHiddenUnits,numFunctions);
allAccuracies = zeros(3,numPercentages,numHiddenUnits,numFunctions);

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
save('metrics_.mat','allPerformances','allAccuracies');
mkdir csv_acc_
mkdir csv_perf_
save4Dcsv(allAccuracies,'csv_acc_',["Train" "Val" "Test"],["80-10-10" "40-20-40" "10-10-80"])
save4Dcsv(allPerformances,'csv_perf_',["Train" "Val" "Test"],["80-10-10" "40-20-40" "10-10-80"])