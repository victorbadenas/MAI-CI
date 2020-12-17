clear all; close all;
MIN_COORD = [1, 1];

Populationsize = 20:20:300;
Generations = 10:20:200;
numGenerations=length(Generations);
numPopulations=length(Populationsize);

results = zeros(numGenerations*numPopulations, 5);

fprintf("-------------------------------------------------------------------------------\n")
fprintf("------------------------------------ START ------------------------------------\n")
fprintf("-------------------------------------------------------------------------------\n")
for i=1:numGenerations
    for j=1:numPopulations
        iteration = (i-1)*numPopulations + j;
        fprintf("now evaluating generations=%i and populations=%i, iteration=%i/%i\n", Generations(i), Populationsize(j), iteration, numGenerations*numPopulations)
        tic
        [x,Fval,vals] = rosenbrock(Generations(i), Populationsize(j));
        time=toc;
        err = vals - MIN_COORD;
        err = err*err';
        results(iteration, :) = [Generations(i), Populationsize(j), err, Fval, time]
        fprintf("-------------------------------------------------------------------------------\n")
    end
end

fprintf("Saving matrices to results.mat\n")
save('metrics.mat','results');

fprintf("saving results to csv\n")
mkdir results
csvwrite("results/results.csv", results);
headers = cellstr(["Generations","Populationsize","Error","Fval","executionTime"]);
csvwrite_with_headers('results/results_headers.csv', results, headers);

fprintf("-------------------------------------------------------------------------------\n")
fprintf("------------------------------------- END -------------------------------------\n")
fprintf("-------------------------------------------------------------------------------\n")