function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));

sum=0;
p= X*theta;
predictions = sigmoid(p);
for  i=1:m
    sum= sum + ((y(i,1)*log(predictions(i,1))) + ((1-y(i,1))*log(1 - predictions(i,1))));
end


J= (-1/m)*(sum);

grad = (1/m)*(((predictions-y)' * X)');


% =============================================================

end
