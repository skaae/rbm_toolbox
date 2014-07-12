function [ dw,db,dc,du,dd,curr_err,chains,chainsy ] = rbmdiscriminative( rbm,v0,ey,opts,chains,chainsy )
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here


[n_hidden,n_visible] = size(rbm.W);
n_samples = size(v0,1);
n_classes = size(rbm.U,2);



n_hidden  = size(rbm.W,1);
n_classes = size(rbm.d,1);
n_samples = size(x,1);


%pre calculate
cwx = repmat(rbm.c,1,n_samples)+rbm.W*x';
F = zeros(n_samples,n_classes);
oyj = zeros(n_samples,n_classes);
for y = 1:n_classes
    rep_U = repmat(rbm.U(:,y),1,n_samples);
    act = cwx + rep_U;
    oyj(:,y) = sum(act,1);
    F(:,y) =  sum( softplus(act ) )+ rbm.d(y);
end
class_normalizer = log_sum_exp_over_rows(F'); 
log_class_prob = F - repmat(class_normalizer',1,n_classes); 


class_prob = exp(log_class_prob);

% if have the class probabilities and the o (not that o changes with each
% iteration of the for loop may move some code inside the )
%  derive update for each param
% move the calculation for each class inside thee 

end

