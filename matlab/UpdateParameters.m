function [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate)
% [W, b] = UpdateParameters(W, b, grad_W, grad_b, learning_rate) computes and returns the
% new network parameters 'W' and 'b' with respect to the old parameters, the
% gradient updates 'grad_W' and 'grad_b', and the learning rate.
for i=1:1:size(W,2)
W{1, i}=W{1, i}-learning_rate*grad_W{1, i};
b{1, i}=b{1, i}-learning_rate*grad_b{1, i};
end

end

