function [nn_params cost ERR MSG] = nnTrain(X, y, layers, num_labels, maxIter, lambda)
	%Returns weights for Neural Networks after training
	
	ERR = 1; MSG = '';

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