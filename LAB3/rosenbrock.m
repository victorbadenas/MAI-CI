%https://es.mathworks.com/matlabcentral/answers/453836-how-can-i-store-the-value-at-each-iteration-of-a-genetic-algorithm

[x, fval, vals] = Rosenbrock  

function [x,Fval,vals] = Rosenbrock 
%FitnessFunction = @(x)(1-x(1))^2+100*(x(2)-x(1)^2)^2;
FitnessFunction = @myFitness;
numberOfVariables = 2;
vals = [];

%opts = gaoptimset('Display', 'none');
opts = gaoptimset('Display', 'off', 'OutputFcn',@my_caca);
%opts = gaoptimset('PlotFcns',{@gaplotbestf,@gaplotdistance});

%--Number of Generations
opts = gaoptimset(opts, 'Generations', 300); % 'StallGenLimit', 50

%--Population Size
opts = gaoptimset(opts, 'PopulationSize', 50);

%--Initial Range
opts = gaoptimset(opts, 'PopInitRange',[-2 -2; 2 2]);

%--Selection
opts = gaoptimset(opts, 'SelectionFcn',@selectiontournament, 'FitnessScalingFcn',@fitscalingprop);

%--Reproduction (crossover and mutation)
%opts = gaoptimset(opts, 'CrossoverFcn',@crossoverscattered);
%opts = gaoptimset(opts, 'MutationFcn',@mutationgaussian);

rng default %rng


record = [];
for n=1:1
    opts = gaoptimset(opts, 'CrossoverFraction', n);
    [x, Fval, exitFlag, Output] = ga(FitnessFunction, numberOfVariables, [], [], [], [], [], [], [], opts);
    record = [record; Fval];
    
    fprintf('The number of generations was : %d\n', Output.generations);
    fprintf('The number of function evaluations was : %d\n', Output.funccount);
    fprintf('The best function value found was : %g\n', Fval);
    formatSpec = 'The best function value was found at point: %7.4f %7.4f \n';
    fprintf(formatSpec,x);
end

function y = myFitness(x)
    y = 100*(x(1)^2-x(2))^2 + (1-x(1))^2;
    vals(1:2) = x;
end

function [state, options,optchanged] =  my_caca(options,state,flag)  
    bananaout(vals, state, flag);
    optchanged = false; 
    %disp(vals); 
end
end

%plot(0:.05:1, record);
%xlabel('Crossover Fraction');
%ylabel('fval');

%Questions:
%Where is the global minimum? Ideal: (a, a^2) -> (1, 1)
%Which is the global minimum? Approx Ideal: 4.70753e-08
%Combination that gives us the best result