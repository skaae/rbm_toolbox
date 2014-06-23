function [dw,db,dc,curr_err,chains] = rbmstatistics(rbm,v0,k,type,chains)
%RBMSTATISTICS collect statistics for RBM
% SEE sections contrastive divergence(CD) and persistent contrastive 
% divergence (PCD), determined by the type parameter     
%
%   INPUTS:
%       rbm       : a rbm struct
%       v0        ; the initial state of the hidden units
%       k         : number of gibbs steps / CD updates
%       type      : PCD|CD persistent CD or CD
%       chains    ; current state of markov chains
%  
%   OUTPUTS
%       dw         : w weights chainge
%       db         : bias of visible layer weight change
%       dc         : bias of hidden layer weight change
%       curr_err   : current squared error
%       chains     : current value of chains
%
%
%CONTRASTIVE DIVERGENCE (TYPE = 'CD')
% Normal contrastive divergence with k CD updates
% 
% See also
%   Hinton, G. (2002). Training Products of Experts by Minimizing Contrastive
%   Divergence. Neural Compu- tation, 14, 1771?1800.
%
%PERSISTENT CONTRASTIVE DIVERGENCE (TYPE = 'PCD')
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
% see also
%     Tieleman, Tijmen. 
%     "Training restricted Boltzmann machines using approximations to the 
%     likelihood gradient." 
%     Proceedings of the 25th international conference on Machine learning. 
%     ACM, 2008.
%
% Copyright Søren Sønderby June 2014

% Collect postivite phase (we already have v0 as imput)
% the positive phase is the same for CD and PCD
h0 = rbmup(rbm,v0,@sigmrnd);  



% collect negative phase by sampling k times
switch type
    case 'CD'
        % For contrastive divergence use the input vectors as starting point
        hid_sampled = h0;
    case 'PCD'
        % for Persistent contrastive divergence we use the persistent chains as
        % starting point for the sampling
        hid_sampled = rbmup(rbm,chains,@sigmrnd);
end
for i=1:k
    vis_sampled       = rbmdown(rbm,hid_sampled,@sigmrnd);
    hid_deterministic = rbmup(rbm,vis_sampled,@sigm);
    hid_sampled       = double(hid_deterministic>rand(size(hid_deterministic)));
end

%update the state of the persistent chains
chains = vis_sampled;

% do not sample last reconstruction as it just introduced sampling noise
vk = vis_sampled;   
hk = hid_deterministic;

% calcualte the postive and negative gradient / aka positive and neg phase
positive_phase = h0' * v0;
negative_phase = hk' * vk;

dw = positive_phase - negative_phase;
db =  sum(v0 - vk)';
dc =  sum(h0 - hk)';
curr_err = sum(sum((v0 - vk) .^ 2));
end

