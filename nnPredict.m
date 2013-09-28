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