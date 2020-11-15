function [performances, accuracies] = trainNN(X,Y,epochs,functions,hiddenUnits,split,nRuns,momentum,lr,gpu)
    performances = zeros(3,nRuns);
    accuracies = zeros(3,nRuns);
    for i = 1:nRuns
        net=feedforwardnet(hiddenUnits);

        for layerIdx=1:(length(net.layers)-1)
            net.layers{layerIdx}.transferFcn = functions.hidden;
        end
        net.layers{end}.transferFcn = functions.output;
        net.performFcn = functions.cost;

        net.trainFcn = 'traingdm';
        net.trainParam.mc = momentum;
        net.trainParam.lr = lr;
        net.trainParam.epochs = epochs;

        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = split.training;
        net.divideParam.valRatio = split.validation;
        net.divideParam.testRatio = split.test;

        net.outputs{end}.processFcns = {};
        for layerIdx=1:(length(net.inputs))
            net.inputs{layerIdx}.processFcns = {'mapminmax'};
        end

        if gpu == 1
            [net,tr,out,E] = train(net,X,Y,'useGPU','yes');
        else
            [net,tr,out,E] = train(net,X,Y);
        end
        performances(:,i)=[tr.best_perf tr.best_vperf tr.best_tperf];
        [~, Yargmax] = max(Y, [], 1);
        [~, Oargmax] = max(out, [], 1);
        accVector = Yargmax == Oargmax;
        
        [~, trainMask] = find(tr.trainMask{1} == 1);
        [~, valMask] = find(tr.valMask{1} == 1);
        [~, testMask] = find(tr.testMask{1} == 1);
        
        trainAcc = sum(accVector(trainMask)) / length(trainMask);
        valAcc = sum(accVector(valMask)) / length(valMask);
        testAcc = sum(accVector(testMask)) / length(testMask);
        accuracies(:,i) = [trainAcc valAcc testAcc];
    end
    performances = mean(performances,2);
    accuracies = mean(accuracies, 2);
end

