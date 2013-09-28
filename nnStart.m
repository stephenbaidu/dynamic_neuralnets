%Read me
%------Files-----------
% fmincg.m
% initWeights.m
% nnCost.m
% nnPredict.m
% nnStart.m
% nnTrain.m
% octave-core.m
% sigmoid.m
% sigmoidGradient.m
%----------------------


%Set configuration, ie: number of nodes in your layers like below
% layers = [no_input_nodes hidden_1 .... hidden_n no_output_nodes];

%Example: layers = [20 10 10 5]; 
% ie:	20 input nodes for layer 1
% 		10 hidden nodes for layer 2
% 		10 hidden nodes for layer 3 
% 		5 output nodes for layer 4
%can also be row vector like [20; 10; 10; 5]



%Explictly set the number of labels/classes
num_labels = 10;

%Set the neural networks layer configuration
layers = [400 25 10]; %As used in Ex4

%Maximum Iteration
maxIter = 10;

%Set lambda for regularization
lambda = 1;

%Using test data given in Ex4
load('ex4data1.mat');

[nn_params cost ERR MSG] = nnTrain(X, y, layers, num_labels, maxIter, lambda);

pred = nnPredict(nn_params, layers, X);

fprintf('\nTraining Set Accuracy: %f %%\n', mean(double(pred == y)) * 100);