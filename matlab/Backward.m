function [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a)
% [grad_W, grad_b] = Backward(W, b, X, Y, act_h, act_a) computes the gradient
% updates to the deep network parameters and returns them in cell arrays
% 'grad_W' and 'grad_b'. This function takes as input:
%   - 'W' and 'b' the network parameters
%   - 'X' and 'Y' the single input data sample and ground truth output vector,
%     of sizes Nx1 and Cx1 respectively
%   - 'act_h' and 'act_a' the network layer pre and post activations when forward
%     forward propogating the input smaple 'X'
N=size(act_a,2);
exp_x=exp(act_a{1,N});
sum_exp=sum(exp_x);
inv_softmax=sum_exp*(exp_x.^-1);
dloss_soft=sum(-Y.*inv_softmax);
dsoft_act_a=((exp_x.*Y)*sum_exp - sum(Y.*exp_x)*exp_x)/(sum_exp*sum_exp);

[m,n]=size(W);
[p,q]=size(b);
grad_W{m,n}= [];
grad_b{p,q}= [];
grad_aa_W{m,n}= [];
grad_aa_b{p,q}= [];

grad_aa_forw_aa{m,n}= [];
grad_L_aa=dloss_soft*dsoft_act_a;
for i=n:-1:1
    %calculate gradient of act_a wrt to input of that layer ie daa_w and
    %daa_b
    if(i>1)
        grad_aa_W{:,i}=transpose(act_a{:,i-1})*(act_a{:,i}.*(1-act_a{:,i}));
        grad_aa_b{:,i}=act_a{:,i}.*(1-act_a{:,i}); 
    else        
        grad_aa_W{:,i}=transpose(X)*(act_a{:,i}.*(1-act_a{:,i}));
        grad_aa_b{:,i}=act_a{:,i}.*(1-act_a{:,i});        
    end
    %calculate gradient dact_a_i+1 wrt act_a_i
    if(i~=n && i~=1)        
        grad_aa_forw_aa{:,i}=act_a{:,i+1}.*act_a{:,i+1}.*W{:,i+1};        
        grad_prop=transpose(sum(grad_prop.*grad_aa_forw_aa{:,i}, 2));        
    elseif(i==n)
        grad_aa_forw_aa{:,i}=grad_L_aa;
        grad_prop=grad_aa_forw_aa{:,i};%grad_prop is dL_acta
    elseif(i==1)        
        grad_aa_forw_aa{:,i}=act_a{:,i+1}.*act_a{:,i+1}.*W{:,i+1}; 
        grad_prop=transpose(sum(grad_prop.*grad_aa_forw_aa{:,i}, 2));  
    end
    %now we update gradients    
    grad_W{:,i}=grad_prop.*grad_aa_W{:,i};
    grad_b{:,i}=grad_prop.*grad_aa_b{:,i};
    
end
end
