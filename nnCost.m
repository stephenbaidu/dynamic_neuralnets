function [J grad] = nnCost(nn_params, layers, num_labels, X, y, lambda)
	%Returns the Cost and Gradient
	J = 0;
	grad = [];
	m = size(X, 1);
	layer_count = size(layers, 1);

	%Converts y into matrix Y where Y(i, :) = [0 1 0 ... 0] for each label 2 example
	if m < 100000	%Implement this only when there are fewer examples say 100000
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
