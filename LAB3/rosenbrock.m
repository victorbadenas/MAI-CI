%https://es.mathworks.com/matlabcentral/answers/453836-how-can-i-store-the-value-at-each-iteration-of-a-genetic-algorithm
function [x,Fval,vals] = Rosenbrock(Generations, PopulationSize, FitnessScalingFcn, SelectionFcn, MutationFcn, CrossoverFcn, PopInitRange) 
    arguments
        Generations {mustBeNumeric, mustBeReal, mustBeInteger} = 300;
        PopulationSize {mustBeNumeric, mustBeReal, mustBeInteger} = 100;
        FitnessScalingFcn = @fitscalingrank;
        SelectionFcn = @selectionstochunif;
        MutationFcn = @mutationgaussian;
        CrossoverFcn = @crossoverscattered;
        PopInitRange (2,2) {mustBeNumeric,mustBeReal} = [-2 -2; 2 2];
    end
    %FitnessFunction = @(x)(1-x(1))^2+100*(x(2)-x(1)^2)^2;
    FitnessFunction = @myFitness;
    numberOfVariables = 2;
    vals = [];

    %opts = gaoptimset('Display', 'none');
%     opts = gaoptimset('Display', 'off', 'OutputFcn',@plotter);
    opts = gaoptimset('Display', 'off');
    %opts = gaoptimset('PlotFcns',{@gaplotbestf,@gaplotdistance});
    
    %--Number of Generations
    opts = gaoptimset(opts, 'Generations', Generations); % 'StallGenLimit', 50

    %--Population Size
    opts = gaoptimset(opts, 'PopulationSize', PopulationSize);

    %--Initial Range
    opts = gaoptimset(opts, 'PopInitRange', PopInitRange);

    %--Selection
    opts = gaoptimset(opts, 'SelectionFcn', SelectionFcn, 'FitnessScalingFcn', FitnessScalingFcn);
    
    %--Reproduction
    opts = gaoptimset(opts, 'MutationFcn', MutationFcn, 'CrossoverFcn', CrossoverFcn);

    rng default %rng


    record = [];
    for n=0:.05:1
        opts = gaoptimset(opts, 'CrossoverFraction', n);
        [x, Fval, exitFlag, Output] = ga(FitnessFunction, numberOfVariables, [], [], [], [], [], [], [], opts);
        record = [record; Fval];

%         fprintf('The number of generations was : %d\n', Output.generations);
%         fprintf('The number of function evaluations was : %d\n', Output.funccount);
%         fprintf('The best function value found was : %g\n', Fval);
%         formatSpec = 'The best function value was found at point: %7.4f %7.4f \n';
%         fprintf(formatSpec,x);
    end

    function y = myFitness(x)
        y = 100*(x(1)^2-x(2))^2 + (1-x(1))^2;
        vals(1:2) = x;
    end

    function [state, options,optchanged] = plotter(options,state,flag)
        bananaout(vals, state, flag);
        optchanged = false; 
    end
end

%plot(0:.05:1, record);
%xlabel('Crossover Fraction');
%ylabel('fval');

%Questions:
%Where is the global minimum? Ideal: (a, a^2) -> (1, 1)
%Which is the global minimum? Approx Ideal: 4.70753e-08
%Combination that gives us the best result