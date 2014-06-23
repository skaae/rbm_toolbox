function [dW,db,dc,curr_err,chains] = pcd(rbm,v0,k,chains)
%PCD Persistent contrastive divergence
%   The PCD approximation is obtained from the CD approximation by replacing the 
%   sample v_k by a sample from a Gibbs chain that is independent from the 
%   sample v_0 of of the training distribution. The algorithm corresponds to
%   standard CD learning without reinitializing the visible units of the Markov 
%   chain with a training sample each time we want to draw a sample v_k
%   approximately from the RBM distribution. Instead one keeps ?persistent? 
%   chains which are run for k Gibbs steps after each parameter update 
%   (i.e., the initial state of the current Gibbs chain is equal to v_k from 
%   the previous update step) 
%
%   INPUTS:
%     rbm: a rbm struct
%     v0        ; the initial state of the hidden units
%     k         :  number of gibbs steps
%     chains    ;  current state of markov chains
%   OUTPUTS
%    dw         : w weights chainge
%    db         : bias of visible layer weight change
%    dc         : bias of hidden layer weight change
%    curr_err   : current squared error
%    chains     : current value of chains
%
%   REFERENCES
%     Tieleman, Tijmen. 
%     "Training restricted Boltzmann machines using approximations to the 
%     likelihood gradient." 
%     Proceedings of the 25th international conference on Machine learning. 
%     ACM, 2008.
% Copyright Søren Sønderby June 2014

% keep h1 for calculation of  of positive phase
h0 = rbmup(rbm,v0,@sigmrnd);  

%update markov chains for k steps
vk_sampled = chains;
for i=1:k
    hk_deterministic = rbmup(rbm,vk_sampled,@sigm);
    hk_sampled = double(hk_deterministic>rand(size(hk_deterministic)));
    vk_sampled       = rbmdown(rbm,hk_sampled,@sigmrnd);
end

%set current state of chains
chains = vk_sampled;

% do not sample last reconstruction as it just introduced sampling noise
vk = vk_sampled;
hk = hk_deterministic;


% calcualte the postive and negative gradient / aka positive and neg phase
positive_phase = h0' * v0;
negative_phase = hk' * vk;

dW = positive_phase - negative_phase;
db =  sum(v0 - vk)';
dc =  sum(h0 - hk)';
curr_err = sum(sum((v0 - vk) .^ 2));
end




