%% CLEAN
clear all; close all;

%% LOAD MATRICES

load("data/Input.mat");
load("data/Output.mat");

%% CONSTANT VALUES

EPOCHS = 1000;
NRUNS = 5;
GPU = 1;
IMAGESDIR = "images/";
layerFcn = struct("hidden","logsig","output","softmax","cost","crossentropy");
split = struct("training",0.8,"validation",0.1,"test",0.1);

trainFcnStrings = ["trainrp","trainscg","traincgb","traincgf","traincgp","trainoss","traingdx","traingdm","traingd"];
hiddenUnits = [50 200 500];

%% INIT PERFORMANCES AND ACCURACIES
allPerformances = zeros(3,length(hiddenUnits),length(trainFcnStrings));
allAccuracies = zeros(3,length(hiddenUnits),length(trainFcnStrings));

%% LOOP FOR ALL TRAIN FUNCTIONS AND ALL HIDDEN UNITS
mkdir(IMAGESDIR);
for i=1:length(trainFcnStrings)
    trainFcn = trainFcnStrings(i);
    mkdir(IMAGESDIR + trainFcn)
    for ih=1:length(hiddenUnits)
        fprintf("Now training with %s trainFcn and %i hiddenUnits\n", trainFcn, hiddenUnits(ih));
        [performances,accuracies,net,tr] = trainNN(Input,Output,EPOCHS,layerFcn,hiddenUnits(ih),split,NRUNS,trainFcn,0,GPU);
        allPerformances(:,ih,i) = performances;
        allAccuracies(:,ih,i) = accuracies;
        
        idString = sprintf("Performance %s fcn %i hiddenUnits",trainFcn,hiddenUnits(ih));
        plot(tr.epoch, tr.perf, tr.epoch, tr.vperf, tr.epoch, tr.tperf);
        legend("training","validation","test");
        title(idString);
        pathString = IMAGESDIR + trainFcn + "/" + hiddenUnits(ih) + ".png";
        exportgraphics(gcf,pathString,"Resolution",300);
        close();
    end
end

%% SAVE RESULT AS MAT
save('metrics_fcnSweep.mat','allPerformances','allAccuracies');

%% SAVE RESULT AS CSV
csvFolder = "csv_acc_trainFcn";
mkdir(csvFolder);
labels = ["train" "val" "test"];
for i=1:length(labels)
    redMat = squeeze(allAccuracies(i,:,:));
    filename = csvFolder + "/" + sprintf("%s.csv", labels(i));
    csvwrite_with_headers(char(filename), redMat, cellstr(trainFcnStrings));
end