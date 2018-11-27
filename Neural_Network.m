% neural network for image recognition

theta = cell(3,1)

SIZE = 29; 
m = 5 % number of training examples
X = ones( SIZE^2 , m);
result = zeros(10,m);
% hidden layers
A_1 = zeros(16,1);
A_2 = zeros(16,1);

a = cell(4,1);

a{1} = X(: , 1);
a{2} = A_1;
a{3} = A_2;
a{4} = A_3;

B = ones(1,3); % bias for all layers

% parameters

theta_0 = ones(16 , SIZE^2 + 1);
theta_1 = ones(16 , 17);
theta_2 = ones(10 , 17);
theta{1} = theta_0;
theta{2} = theta_1;
theta{3} = theta_2;

% Just call the learn function with 
% appropriate data along with learning parameters 
% to make the network learn

function theta = learn(t , a , X , result , alpha , maxiter)

for g = 1:maxiter 
    for j = 3:1 % go backwards
        t{j+1} = t{j+1} - alpha*backprop(j , t , a , result , X);
    end
end

theta = t;
end

function J_j = backprop(j, theta , a, result , X)

J_j = zeros(size(theta{j+1}));

for i=1:size(theta{j+1} , 1)
    for k=1:(size(theta{j+1} , 2) - 1)
        J_prime_f = 0;
        
        % for theta j,i,k
        for p=1:m
            a{1} = X(: , i);
            [a{2} , a{3} , ~ ] = forwardpropagation(a{1} , theta{1} , theta{2} , theta{3} , B);
            y = result(:,i);
            % change a_j and y by using forward propogation ?
            J_prime_f = J_prime_f + cost_prime(i,k,theta{j+1},a{j+1},y);
        end
        J_prime_f = J_prime_f / m;
        J_j(i,k) = J_prime_f;
    end
end

end

function J_prime = cost_prime(i,k,theta_j,a_j,y)
h = sigmoid(theta_j*a_j);
J_prime = 2*(h(i)-y(i))*sigmoid_prime(theta_j(i,:)*a_j)*a_j(k);
end

function y = sigmoid_prime(z)
y = exp(z)/(exp(z) + ones(size(z))).^2;
end

function J = cost( h , y)
J = sum((h-y).^2)/(2 * size(h , 1));
end

function [a_1 , a_2 , y] = forwardpropagation(x,t_0 , t_1 , t_2 , B)

x = [B(1) ; x];
a_1 = sigmoid(t_0*x);
a_1 = [B(2) ; a_1];
a_2 = sigmoid(t_1*a_1);
a_2 = [B(3) ; a_2];
y = sigmoid(t_2*a_2);

end

function y = sigmoid(z)

y = 1./(exp( -1 * z) + ones(size(z)));

end