function [grads,curr_err,chains,chainsy] = rbmgenerative(rbm,v0,ey,opts,chains,chainsy)
%RBMGENERATIVE calculate weight updates for generative RBM
% SEE sections contrastive divergence(CD) and persistent contrastive
% divergence (PCD), determined by opts.traintype
%
%   INPUTS:
%       rbm       : a rbm struct
%       v0        : the initial state of the hidden units
%       ey        : one hot encoded labels if classRBM otherwise empty
%       opts      : opts struct
%       chains    : current state of markov chains for visible units
%       chainsy   : current state of markov chains for label visible units
%
%   OUTPUTS
%      A grads struct with the fields:
%       grads.dw   : w weights chainge normalized by minibatch size
%       grads.db   : bias of visible layer weight change norm by minibatch size
%       grads.dc   : bias of hidden layer weight change norm by minibatch size
%       grads.du   : class label layer weight change norm by minibatch size
%       grads.dd   : class label hidden bias weight change norm by minibatch size
%       curr_err   : current squared error normalized by minibatch size
%       chains     : updated value of chains for visible units
%       chainsy    : updated value of chains for label visible units.
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

%% add dropout
down = @rbmdown;
if rbm.dropout_hidden > 0
    up = @(rbm, vis,ey,act_func) rbmup(rbm, vis,ey,act_func).*rbm.hidden_mask;
    
else
    up = @rbmup;
end

h0 = up(rbm,v0,ey,@sigmrnd);

% For contrastive divergence use the input vectors as starting point
% for Persistent contrastive divergence we use the persistent chains as
% starting point for the sampling
switch type
    case 'CD'
        hid = h0;
    case 'PCD'
        hid= up(rbm,chains,chainsy,@sigmrnd);
end

for drop_out_mask = 1:(opts.cdn - 1)
    [visx, visy] = down(rbm,hid,@sigmrnd);
    hid = up(rbm,visx,visy,@sigmrnd);
end

% in last up/down dont sample hidden because it introduces sampling noise
[vk, vky] = down(rbm,hid,@sigmrnd);
hk        = up(rbm,vk,vky,@sigm);

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
if rbm.classRBM == 1
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

grads.dw = dw;
grads.db = db;
grads.dc = dc;
grads.du = du;
grads.dd = dd;

end

