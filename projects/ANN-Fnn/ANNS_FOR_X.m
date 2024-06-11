% Load the data
[alphabets, targets] = prprob;

% Initialize variables
inputSize = size(alphabets, 1); % Change
hiddenSize = 20; % Increase the number of neurons in the hidden layer
outputSize = size(targets, 1);
learningRate = 0.005; % Reduce learning rate

% Initialize weights and biases
W1 = rand(hiddenSize, inputSize) * 0.1 - 0.05;
b1 = rand(hiddenSize, 1) * 0.1 - 0.05;
W2 = rand(outputSize, hiddenSize) * 0.1 - 0.05;
b2 = rand(outputSize, 1) * 0.1 - 0.05;

% Activation function
sigmoid = @(x) 1 ./ (1 + exp(-x));

% Training
for epoch = 1:500
    % Split data for cross-validation
    cv = cvpartition(size(alphabets, 2), 'KFold', 5);
    for i = 1:cv.NumTestSets
        trainIdx = cv.training(i);
        testIdx = cv.test(i);
        
        % Training using training data only
        for z = find(trainIdx)' 
            % Forward propagation
            
            a1 = alphabets(:, z);
            z2 = W1 * a1 + b1;
            a2 = sigmoid(z2);
            z3 = W2 * a2 + b2;
            a3 = sigmoid(z3);

            % Error
            delta3 = a3 - targets(:, z); % Change
            delta2 = (W2' * delta3) .* a2 .* (1 - a2);

            % Update weights and biases
            W2 = W2 - learningRate * delta3 * a2';
            b2 = b2 - learningRate * delta3;
            W1 = W1 - learningRate * delta2 * a1';
            b1 = b1 - learningRate * delta2;
        end
        
        % Evaluate performance using test data
        for z = find(testIdx)' 
            % Forward propagation
            a1 = alphabets(:, z);
            z2 = W1 * a1 + b1;
            a2 = sigmoid(z2);
            z3 = W2 * a2 + b2;
            a3 = sigmoid(z3);
            
            % Calculate error
            % ... (Code to calculate error can be added here)
        end
    end
end

% Test the network
J = alphabets(:,26); 
output = sigmoid(W2 * sigmoid(W1 * J + b1) + b2);
[~, answer] = max(output);
figure; plotchar(alphabets(:, answer));

% Test the network with noisy 'J'
noisyJ = J + randn(35,1) * 0.2;
figure; plotchar(noisyJ);
output2 = sigmoid(W2 * sigmoid(W1 * noisyJ + b1) + b2);
[~, answer2] = max(output2);
figure; plotchar(alphabets(:, answer2));
