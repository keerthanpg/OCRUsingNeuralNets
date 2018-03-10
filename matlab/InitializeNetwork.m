function [W, b] = InitializeNetwork(layers)
% InitializeNetwork([INPUT, HIDDEN, OUTPUT]) initializes the weights and biases
% for a fully connected neural network with input data size INPUT, output data
% size OUTPUT, and HIDDEN number of hidden units.
% It should return the cell arrays 'W' and 'b' which contain the randomly
% initialized weights and biases for this neural network.
m=size(layers, 2);
W = {};
b = {};
Navg=(layers(1,1)+layers(1,m))/2;
for i=1:1:m-1
    W{1,i}=normrnd(0,1/Navg,[layers(1,i) layers(1,i+1)]);
    b{1,i}=zeros(1,layers(1,i+1));
end

end
