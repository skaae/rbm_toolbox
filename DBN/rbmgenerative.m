function [grads,curr_err,chains,chainsy] = rbmgenerative(rbm,v0,ey,opts,chains,chainsy,debug)
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
%       debug     : if it exists and is 1 save intermediate values to file
%                   currentworkdir/test_rbmgenerative.mat
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
% See
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
% See also RBMDISCRMINATIVE RBMHYBRID RBMSEMISUPLEARN 
%
% Copyright Søren Sønderby June 2014

type = opts.traintype;

% add dropout
if rbm.dropout_hidden > 0
    up = @(rbm, vis,ey,act_func) rbm.rbmup(rbm, vis,ey,act_func).*rbm.hidden_mask;    
else
    up = @rbm.rbmup;
end

h0 = up(rbm,v0,ey,@sigm);   % hidden positive statistic larochelle does not sample postive statistics? 
h0_rnd =  double(h0 > rbm.rand(size(h0)));

% For contrastive divergence use the input vectors as starting point
% for Persistent contrastive divergence we use the persistent chains as
% starting point for the sampling
switch type
    case 'CD'
        hid = h0_rnd;
    case 'PCD'
        hid= up(rbm,chains,chainsy,@sigmrnd);
end

for n = 1:(opts.cdn - 1)
    visx = rbmdownx(rbm,hid,@sigmrnd);
    visy = rbm.rbmdowny(rbm,hid);
    hid = up(rbm,visx,visy,@sigmrnd);
end

% in last up/down dont sample hidden because it introduces sampling noise
vkx   = rbmdownx(rbm,hid,@sigmrnd);   
vky   = rbm.rbmdowny(rbm,hid);
vky   = samplematrix(vky); % sample visible state

hk   = up(rbm,vkx,vky,@sigm);         

% debugging
if exist('debug','var') && debug == 1
    warning('Debugging rbmgenerative')
    vkx_sigm   = rbmdownx(rbm,hid,@sigm);   % debugging
    vky_prob   = rbm.rbmdowny(rbm,hid); 
    hk_sample  = up(rbm,vkx,vky,@sigmrnd); 
    save('test_rbmgenerative','h0','h0_rnd','hk_sample',...
                 'hk','v0','vkx','vkx_sigm','vky','vky_prob');
end

%update the state of the persistent chains if PCD othwise return empty chains
switch type
    case 'PCD'
        chains = vkx;
        chainsy = vky;
    case 'CD'
        chains = [];
        chainsy = [];
end

%% calculate gradients
% h0  : postivie statistic for hidden units
% v0  : positive statistic for the visible units
% vk  : negative stat for visible units 
% vky : negative stat for label visible units
% hk  : negative stat for hidden units

% calcualte the postive and negative gradient / aka positive and neg phase
positive_phase = h0' * v0;
negative_phase = hk' * vkx;

dw = positive_phase - negative_phase;
db =  sum(v0 - vkx,1)';
dc =  sum(h0 - hk,1)';

% normalize by minibatch size
dw = dw / opts.batchsize;
db = db / opts.batchsize;
dc = dc / opts.batchsize;

% for hinton DBN update bias and variables for du and dd
if rbm.classRBM == 1
    positive_phasey = h0' * ey;
    negative_phasey = hk' * vky;
    du = positive_phasey - negative_phasey;
    dd = sum(ey - vky,1)';
    du = du / opts.batchsize;
    dd = dd/ opts.batchsize;
else
    % return zero gradients for non cRBM's
    du = rbm.zeros(size(rbm.U));
    dd = rbm.zeros(size(rbm.d));
end

curr_err = sum(sum((v0 - vkx) .^ 2)) / opts.batchsize;

grads.dw = dw;
grads.db = db;
grads.dc = dc;
grads.du = du;
grads.dd = dd;

end

