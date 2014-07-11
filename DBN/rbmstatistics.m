function [dw,db,dc,du,dd,curr_err,chains,chainsy] = rbmstatistics(rbm,v0,ey,opts,chains,chainsy,avg)
%RBMSTATISTICS collect statistics for RBM and calculate weight changes
% SEE sections contrastive divergence(CD) and persistent contrastive
% divergence (PCD), determined by the type parameter
%
%   INPUTS:
%       rbm       : a rbm struct
%       v0        : the initial state of the hidden units
%       ey        : one hot encoded labels if hintonDBN otherwise empty
%       opts      : opts struct
%       chains    : current state of markov chains
%
%   OUTPUTS
%       dw         : w weights chainge normalized by minibatch size
%       db         : bias of visible layer weight change norm by minibatch size
%       dc         : bias of hidden layer weight change norm by minibatch size
%       du         : class label layer weight change norm by minibatch size
%       dd         : class label hidden bias weight change norm by minibatch size
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
% NOTATION
% data  : all data given as      [n_samples   x #vis]
%    v  : all data given as      [n_samples   x #vis]
%   ey  : all data given as      [n_samples   x #n_classes]
%    W  : vis - hid weights      [ #hid       x #vis ]
%    U  : label - hid weights    [ #hid       x #n_classes ]
%    b  : bias of visible layer  [ #vis       x 1]
%    c  : bias of hidden layer   [ #hid       x 1]
%    d  : bias of label layer    [ #n_classes x 1]
%
% Copyright Søren Sønderby June 2014

% Collect postivite phase (we already have v0 as imput)
% the positive phase is the same for CD and PCD
type = opts.traintype;

h0 = rbmup(rbm,v0,ey,@sigmrnd);

% For contrastive divergence use the input vectors as starting point
% for Persistent contrastive divergence we use the persistent chains as
% starting point for the sampling
switch type
    case 'CD'
        hid = h0;
    case 'PCD'
        hid= rbmup(rbm,chains,chainsy,@sigmrnd);
end

for i = 1:(opts.cdn - 1)
    [visx, visy] = rbmdown(rbm,hid,@sigmrnd);
     hid = rbmup(rbm,visx,visy,@sigmrnd);
end

% in last up/down dont sample hidden because it introduces sampling noise
[vk, vky] = rbmdown(rbm,hid,@sigmrnd);
 hk       = rbmup(rbm,vk,vky,@sigm);

%update the state of the persistent chains if PCD othwise return empty chains
switch type
    case 'PCD'
        chains = vk;
        chainsy = vky;
    case 'CD'
        chains = [];
        chainsy = [];
end


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

% for hinton DBN update bias and variables for du and dd
if rbm.hintonDBN == 1
    positive_phasey = h0' * ey;
    negative_phasey = hk' * vky;
    du = positive_phasey - negative_phasey;
    dd = sum(ey - vky)';
    du = du / opts.batchsize;
    dd = dd/ opts.batchsize;
else
    du = [];
    dd = [];
end

curr_err = sum(sum((v0 - vk) .^ 2)) / opts.batchsize;

end

