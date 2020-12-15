%https://es.mathworks.com/help/optim/ug/banana-function-minimization.html

fun = @(x)(100*(x(2) - x(1)^2)^2 + (1 - x(1))^2); 
x0 = [-1.9,2]; 

%options = optimset('OutputFcn',@bananaout,'Display','off'); 
%[x,fval,eflag,output] = fminsearch(fun,x0,options); 
%title 'Rosenbrock solution via fminsearch'

%Fcount = output.funcCount; 
%disp(['Number of function evaluations for fminsearch was ',num2str(Fcount)])
%disp(['Number of solver iterations for fminsearch was ',num2str(output.iterations)])

options = optimoptions('fminunc','Display','off','OutputFcn',@bananaout,'Algorithm','quasi-newton'); 
[x,fval,eflag,Output] = fminunc(fun,x0,options); 
title 'Rosenbrock solution via fminunc'

fprintf('The number of generations was : %d\n', Output.iterations);
fprintf('The number of function evaluations was : %d\n', Output.funcCount);
fprintf('The best function value found was : %g\n', fval);
formatSpec = 'The best function value was found at point: %7.4f %7.4f \n';
fprintf(formatSpec,x);
