function [performances,accuracies,bestNet,bestTr] = trainNN(X,Y,epochs,functions,hiddenUnits,split,nRuns,trainFcn,trainFcnParams,gpu)
    performances = zeros(3,nRuns);
    accuracies = zeros(3,nRuns);
    bestAcc = 0;
    for i = 1:nRuns
        net = feedforwardnet(hiddenUnits);

        for layerIdx=1:(length(net.layers)-1)
            net.layers{layerIdx}.transferFcn = functions.hidden;
        end
        net.layers{end}.transferFcn = functions.output;
        net.performFcn = functions.cost;

        net.trainFcn = trainFcn;
        
        if trainFcnParams ~= 0
            trainFcnParamsNames = fieldnames(trainFcnParams);
            for idx = 1:length(trainFcnParamsNames)
                net.trainParam.(trainFcnParamsNames{idx}) = trainFcnParams.(trainFcnParamsNames{idx});
            end
        end

        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = split.training;
        net.divideParam.valRatio = split.validation;
        net.divideParam.testRatio = split.test;

        net.outputs{end}.processFcns = {};
        for layerIdx=1:(length(net.inputs))
            net.inputs{layerIdx}.processFcns = {};
        end

        if gpu == 1
            [net,tr,out,~] = train(net,X,Y,'useGPU','yes');
        else
            [net,tr,out,~] = train(net,X,Y);
        end
        performances(:,i)=[tr.best_perf tr.best_vperf tr.best_tperf];
        [~, Yargmax] = max(Y, [], 1);
        [~, Oargmax] = max(out, [], 1);
        accVector = Yargmax == Oargmax;
        
        trainAcc = sum(accVector(tr.trainInd)) / length(tr.trainInd);
        valAcc = sum(accVector(tr.valInd)) / length(tr.valInd);
        testAcc = sum(accVector(tr.testInd)) / length(tr.testInd);

        accuracies(:,i) = [trainAcc valAcc testAcc];
        
        if (valAcc > bestAcc) || (i == 1)
            bestTr = tr;
            bestNet = net;
        end
    end
    performances = mean(performances,2);
    accuracies = mean(accuracies, 2);
end

