function [dW,db,dc,curr_err] = cdk(rbm,v0,k )
%CDK performs contrastive divergence n_times
%   Contrastive divergence sampling.
%   see "A practical guide to training restricted Boltzmann machines" section
%   3.4.
%   INPUTS:
%     rbm: a rbm struct
%     v1 ; the initial state of the hidden units
%     n  :  number of contrastive divergence steps
%   OUTPUTS
%     dW, db,dc: weight and visible/hidden biases
%     curr_err : current error
% See also
%   Hinton, G. (2002). Training Products of Experts by Minimizing Contrastive
%   Divergence. Neural Compu- tation, 14, 1771?1800.

% keep h1 for calculation of  of positive phase
h0 = rbmup(rbm,v0,@sigmrnd);  
hid_sampled = h0;  
for i=1:k
    vis_sampled       = rbmdown(rbm,hid_sampled,@sigmrnd);
    hid_deterministic = rbmup(rbm,vis_sampled,@sigm);
    hid_sampled       = double(hid_deterministic>rand(size(hid_deterministic)));
end
vk = vis_sampled;   

% do not sample last reconstruction as it just introduced sampling noise
hk = hid_deterministic;

% calcualte the postive and negative gradient / aka positive and neg phase
positive_phase = h0' * v0;
negative_phase = hk' * vk;

dW = positive_phase - negative_phase;
db =  sum(v0 - vk)';
dc =  sum(h0 - hk)';
curr_err = sum(sum((v0 - vk) .^ 2));
end

