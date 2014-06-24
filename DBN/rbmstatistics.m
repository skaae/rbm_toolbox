function [dw,db,dc,curr_err,chains] = rbmstatistics(rbm,v0,opts,type,chains)
%RBMSTATISTICS collect statistics for RBM and calculate weight changes
% SEE sections contrastive divergence(CD) and persistent contrastive
% divergence (PCD), determined by the type parameter
%
%   INPUTS:
%       rbm       : a rbm struct
%       v0        : the initial state of the hidden units
%       opts      : opts struct
%       type      : PCD|CD persistent CD or CD
%       chains    _ current state of markov chains
%
%   OUTPUTS
%       dw         : w weights chainge normalized by minibatch size
%       db         : bias of visible layer weight change norm by minibatch size
%       dc         : bias of hidden layer weight change norm by minibatch size
%       curr_err   : current squared error normalized by minibatch size
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



% For contrastive divergence use the input vectors as starting point
% for Persistent contrastive divergence we use the persistent chains as
% starting point for the sampling
switch type
    case 'CD'
        hid = h0;
    case 'PCD'
        hid= rbmup(rbm,chains,@sigmrnd);
end

vk = rbmdown(rbm,hid,@sigmrnd);
hk = rbmup(rbm,vis_sampled,@sigmrnd);

%update the state of the persistent chains
chains = vk;

% calcualte the postive and negative gradient / aka positive and neg phase
positive_phase = h0' * v0;
negative_phase = hk' * vk;

dw = positive_phase - negative_phase;
db =  sum(v0 - vk)';
dc =  sum(h0 - hk)';

% normalize by minibatch size
dw = dw / opts.batchsize;
db = db / opts.batchsize;
dc = dc / opts.batchsize;
curr_err = sum(sum((v0 - vk) .^ 2)) / opts.batchsize;
end

