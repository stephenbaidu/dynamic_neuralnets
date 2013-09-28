I've modified the Neural Networks code to work for any specified layer configuration.  
Example: layers = [400 25 25 10]; can be passed to the algorithm and it will do the rest. No need to change code when you have to change your configuration.  

Each file starts with the **name in bold.**


**nnStart.m** The file to run  
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
  
  

**nnTrain.m**  
function [nn_params cost ERR MSG] = nnTrain(X, y, layers, num_labels, maxIter, lambda)  
	%Returns weights for Neural Networks after training  
	  
	ERR = 1;  
  
	layer_count = max(size(layers));  
	if size(layers, 1) == 1 %Convert column vectors to row vector  
		layers = layers';  
	end  
	%Make sure labels in y start from 1 to num_labels(number of output nodes)  
	if (min(y) ~= 1) | (max(y) ~= num_labels)  
		MSG = 'Labels/classes should start from 1 to number of labels/classes';  
		return;  
	end  
  
	initial_nn_params = [];  
	for l = 1:layer_count - 1  
		r = layers(l + 1);  
		c = layers(l) + 1;  
		initial_nn_params = [initial_nn_params; initWeights(r, c)(:)];  
	end  
  
	options = optimset('MaxIter', maxIter);  
	costFunction = @(p) nnCost(p, layers, num_labels, X, y, lambda);  
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);  
	ERR = 0;  
end  
  
  

**nnCost.m**  

function [J grad] = nnCost(nn_params, layers, num_labels, X, y, lambda)  
	%Returns the Cost and Gradient  
	J = 0;  
	grad = [];  
	m = size(X, 1);  
	layer_count = size(layers, 1);  
  
	%Converts y into matrix Y where Y(i, :) = [0 1 0 ... 0] for each label 2 example  
	if m < 100000	%Implement this only when there are fewer examples  
		Y = eye(num_labels)(y, :);  
	else  
		Y = zeros(m, num_labels);  
		for i = 1:m  
			Y(i, y(i)) = 1;  
		end  
	end  
  
	%Reshape all theta  
	Theta = cell(layer_count - 1, 1);  
	Theta_grad = cell(layer_count-1, 1);  
	for l = 1:layer_count - 1  
		r = layers(l + 1);  
		c = layers(l) + 1;  
		start_index = layers(2:l)' * (layers(1:l-1) + 1) + 1;  
		stop_index = start_index  + (r * c) - 1;  
		Theta{l} = reshape(nn_params(start_index:stop_index), r, c);  
		Theta_grad{l} = zeros(size(Theta{l}));  
	end  
  
	%Compute activation units  
	H = X;  
	for l = 1:layer_count - 1  
		H = [ones(m, 1) H];  
		H = sigmoid(H * Theta{l}');  
	end  
  
	for l = 1:layer_count - 1  
		J = J + sum(sum(Theta{l}(:, 2:end) .* Theta{l}(:, 2:end)));  
	end  
	J = (-1/m) * (sum(sum((Y .* log(H)) + ((1 - Y) .* (log(1 - H)))))) + (lambda/(2 * m)) * J;  
  
	A = cell(layer_count, 1);  
	Z = cell(layer_count, 1);  
	D = cell(layer_count, 1);  
  
	for t = 1:m  
		%---------Forward Propagation-----------  
		for l = 1:layer_count  
			if l == 1  
				A{l} = [1; X(t,:)']; %Assign training example to layer 1  
			elseif l == layer_count  
				Z{l} = Theta{l-1} * A{l-1};  
				A{l} = sigmoid(Z{l});  
			else  
				Z{l} = Theta{l-1} * A{l-1};  
				A{l} = [1; sigmoid(Z{l})];  
			end  
		end  
  
		%---------Backward Propagation----------		  
		for l = fliplr(2:layer_count)  
			if l == layer_count  
				D{l} = A{l} - Y(t,:)';  
			else  
				D{l} = (Theta{l}' * D{l+1})(2:end) .* sigmoidGradient(Z{l});  
				% D{l} = (Theta{l}' * D{l+1}) .* [1; sigmoidGradient(Z{l})];  
				% D{l} = D{l}(2:end);  
			end  
		end  
  
		%---------Big delta update--------------  
		for l=1:layer_count - 1  
			Theta_grad{l} = Theta_grad{l} + D{l+1} * A{l}';  
		end  
  
	end  
  
	%------------Update gradients---------------  
	for l=1:layer_count - 1  
		Theta_grad{l} = (1/m) * Theta_grad{l} + (lambda/m) * [zeros(size(Theta{l}, 1), 1) Theta{l}(:,2:end)];  
	end  
  
	%------------Unroll gradients---------------  
	for l=1:layer_count - 1  
		grad = [grad; Theta_grad{l}(:)];  
	end  
end  
  
  

**nnPredict.m**  
function p = nnPredict(nn_params, layers, X)  
	m = size(X, 1);  
	p = zeros(m, 1);  
	layer_count = max(size(layers));  
	if size(layers, 1) == 1 %Convert column vectors to row vector  
		layers = layers';  
	end  
	num_labels = layers(layer_count);  
	  
	%Reshape all theta  
	Theta = cell(layer_count - 1, 1);  
	for l = 1:layer_count - 1  
		r = layers(l + 1);  
		c = layers(l) + 1;  
		start_index = layers(2:l)' * (layers(1:l-1) + 1) + 1;  
		stop_index = start_index  + (r * c) - 1;  
		Theta{l} = reshape(nn_params(start_index:stop_index), r, c);  
	end  
  
	H = X;  
	for l = 1:layer_count - 1  
		H = sigmoid([ones(m, 1) H] * Theta{l}');  
	end  
	[dummy, p] = max(H, [], 2);  
end  

  
  
**initWeights.m**  
function W = initWeights(row_count, col_count)  
	%Randomly Initialize Weights  
	epsilon_init = 0.12;  
	W = rand(row_count, col_count) * 2 * epsilon_init - epsilon_init;  
end    
  
**The following can be copied from Ex4**  

* fmincg.m
* octave-core.m
* sigmoid.m
* sigmoidGradient.m