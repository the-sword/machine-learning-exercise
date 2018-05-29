function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m,1) X];

%%==========forward propagation===========
a2 = [ones(m,1) (sigmoid(Theta1 * X'))'];
h_theta = sigmoid(Theta2 * a2');
%disp(size(h_theta))
yk = zeros(num_labels,m);
for i= 1:m
   yk(y(i),i) = 1; 
end
%each feature sum
sum_each_fea = sum((-yk).*log(h_theta)-(1-yk).*log(1-h_theta));
%all feature sum
J = 1/m * sum(sum_each_fea);
    
%==========Regularized cost function==========
theta1Zero = Theta1;
theta2Zero = Theta2;

theta1Zero(:,1) = 0;
theta2Zero(:,1) = 0;

regu_cost = lambda/( 2 * m)*( sum(sum(theta1Zero.^2)) + sum(sum(theta2Zero.^2)));

J = J + regu_cost;

%==========Backpropagation==========
for t= 1:m
    %forward pro
    a1 = X(t,:);    %1*401
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);   %25*1
    
    a2 = [1;a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    %backward pro
    z2 = [1;z2];
    delta_3 = a3 - yk(:,t); %10*1
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); 
    %skip delta_2(zero)
    delta_2 = delta_2(2:end);   %25*1
    
    %disp(size(a1))
    Theta2_grad = Theta2_grad + delta_3 * a2';
    Theta1_grad = Theta1_grad + delta_2 * a1;   %a1 need not Transpose
  
end

%Theta1_grad = Theta1_grad ./ m; %25*401
%Theta2_grad = Theta2_grad ./ m;

% Regularization
Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
