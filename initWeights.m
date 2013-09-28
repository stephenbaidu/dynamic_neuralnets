function W = initWeights(row_count, col_count)
	%Randomly Initialize Weights
	epsilon_init = 0.12;
	W = rand(row_count, col_count) * 2 * epsilon_init - epsilon_init;
end