%%
% number of samples of each class
N = 20;
% define inputs and outputs
offset = 5; % offset for second class
x = [randn(2,N) randn(2,N)+offset]; % infigure(1)
plotpc(net.IW{1},net.b{1});
% Plot a classification line on a perceptron vector plotputs
y = [zeros(1,N) ones(1,N)]; % outputs
% Plot input samples with plotpv (Plot perceptron input/target vectors)
figure(1)
plotpv(x,y);

%%

net = perceptron;
% Take a look to the default parameters of this perceptron. Notice that the performance/cost
%function used is mae (mean absolute error), the training function is trainc (trains a network
%with weight and bias learning rules with incremental updates) and the transfer function
%used is hardlim (Hard-limit).
net = train(net, x, y);
view(net);

%%

figure(1)
plotpc(net.IW{1},net.b{1});
% Plot a classification line on a perceptron vector plot