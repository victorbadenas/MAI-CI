clear all; close all;
diary log.log
diary on
MIN_COORD = [1, 1];

% rank / proportional
% uniform / roulette / tournament
% gaussian / uniform
% intermediate recombination
% single point / two point crossover  (?) / intermediate / arithmetic
% parameters related to the convrgence

Generations = 50:20:90;
Populationsize = 240:20:300;
FitnessScalingFcns = {@fitscalingrank, @fitscalingprop};
SelectionFcns = {@selectiontournament, @selectionuniform, @selectionroulette};
MutationFcns = {@mutationgaussian, @mutationuniform};
CrossoverFcns = {@crossoversinglepoint, @crossovertwopoint, @crossoverintermediate, @crossoverarithmetic};

elements = {Populationsize, Generations, 1:length(FitnessScalingFcns), 1:length(SelectionFcns), 1:length(MutationFcns), 1:length(CrossoverFcns)}; %cell array with N vectors to combine
combinations = cell(1, numel(elements)); %set up the varargout result
[combinations{:}] = ndgrid(elements{:});
combinations = cellfun(@(x) x(:), combinations,'uniformoutput',false); %there may be a better way to do this
parameterCombinations = [combinations{:}]; % NumberOfCombinations by N matrix. Each row is unique.

results = zeros(length(parameterCombinations), length(elements)+3);

fprintf("-------------------------------------------------------------------------------\n")
fprintf("------------------------------------ START ------------------------------------\n")
fprintf("-------------------------------------------------------------------------------\n")

%% TESTS
fprintf("\n------------------------------------ TESTS ------------------------------------\n")
params = parameterCombinations(1, :);
for i=1:length(FitnessScalingFcns)
    fprintf("FitnessScalingFcns iteration=%i/%i\n", i, length(FitnessScalingFcns));
    params(3) = i;
    FitnessScalingFcn = FitnessScalingFcns{params(3)};
    SelectionFcn = SelectionFcns{params(4)};
    MutationFcn = MutationFcns{params(5)};
    CrossoverFcn = CrossoverFcns{params(6)};
    [x,Fval,vals] = rosenbrock(params(1), params(2), FitnessScalingFcn, SelectionFcn, MutationFcn, CrossoverFcn);
end

params = parameterCombinations(1, :);
for i=1:length(SelectionFcns)
    fprintf("SelectionFcns iteration=%i/%i\n", i, length(SelectionFcns));
    params(4) = i;
    FitnessScalingFcn = FitnessScalingFcns{params(3)};
    SelectionFcn = SelectionFcns{params(4)};
    MutationFcn = MutationFcns{params(5)};
    CrossoverFcn = CrossoverFcns{params(6)};
    [x,Fval,vals] = rosenbrock(params(1), params(2), FitnessScalingFcn, SelectionFcn, MutationFcn, CrossoverFcn);
end

params = parameterCombinations(1, :);
for i=1:length(MutationFcns)
    fprintf("MutationFcns iteration=%i/%i\n", i, length(MutationFcns));
    params(5) = i;
    FitnessScalingFcn = FitnessScalingFcns{params(3)};
    SelectionFcn = SelectionFcns{params(4)};
    MutationFcn = MutationFcns{params(5)};
    CrossoverFcn = CrossoverFcns{params(6)};
    [x,Fval,vals] = rosenbrock(params(1), params(2), FitnessScalingFcn, SelectionFcn, MutationFcn, CrossoverFcn);
end

params = parameterCombinations(1, :);
for i=1:length(CrossoverFcns)
    fprintf("CrossoverFcns iteration=%i/%i\n", i, length(CrossoverFcns));
    params(6) = i;
    FitnessScalingFcn = FitnessScalingFcns{params(3)};
    SelectionFcn = SelectionFcns{params(4)};
    MutationFcn = MutationFcns{params(5)};
    CrossoverFcn = CrossoverFcns{params(6)};
    [x,Fval,vals] = rosenbrock(params(1), params(2), FitnessScalingFcn, SelectionFcn, MutationFcn, CrossoverFcn);
end

%% EXPERIMENTS
fprintf("--------------------------------- EXPERIMENTS ---------------------------------\n")
for i=1:length(parameterCombinations)
    parameters = parameterCombinations(i, :);
    FitnessScalingFcn = FitnessScalingFcns{parameters(3)};
    SelectionFcn = SelectionFcns{parameters(4)};
    MutationFcn = MutationFcns{parameters(5)};
    CrossoverFcn = CrossoverFcns{parameters(6)};
    fprintf("now evaluating generations=%i and populations=%i, FitnessScalingFcn=%i,\n\t SelectionFcn=%i, MutationFcn=%i, CrossoverFcn=%i,\n\t iteration=%i/%i\n", parameters(1), parameters(2), parameters(3), parameters(4), parameters(5), parameters(6), i, length(parameterCombinations))
    tic
    [x,Fval,vals] = rosenbrock(parameters(1), parameters(2), FitnessScalingFcn, SelectionFcn, MutationFcn, CrossoverFcn);
    time=toc;
    err = vals - MIN_COORD;
    err = err*err';
    tmp = [parameters, err, Fval, time]
    results(i, :) = tmp;
    fprintf("-------------------------------------------------------------------------------\n")
end

fprintf("Saving matrices to results.mat\n")
save('metrics.mat','results');

fprintf("saving results to csv\n")
mkdir results
csvwrite("results/results.csv", results);
headers = cellstr(["Generations", "Populationsize", "FitnessScalingFcns", "SelectionFcns", "MutationFcns", "CrossoverFcns", "error", "Fval", "time"]);
csvwrite_with_headers('results/results_headers.csv', results, headers);

fprintf("-------------------------------------------------------------------------------\n")
fprintf("------------------------------------- END -------------------------------------\n")
fprintf("-------------------------------------------------------------------------------\n")
diary off