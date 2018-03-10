function [accuracy, loss] = ComputeAccuracyAndLoss(W, b, data, labels)
% [accuracy, loss] = ComputeAccuracyAndLoss(W, b, X, Y) computes the networks
% classification accuracy and cross entropy loss with respect to the data samples
% and ground truth labels provided in 'data' and labels'. The function should return
% the overall accuracy and the average cross-entropy loss.
outputs=Classify(W,b,data);
[D,N]=size(data);
[M,I] = max(outputs);
[P,Q] = max(transpose(labels));
loss=sum(-Q.*log(M));
accuracy=(sum(I==Q))/size(labels,1);
end
