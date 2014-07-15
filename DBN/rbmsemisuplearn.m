function [ grads,curr_err,chains_comb,chainsy_type ] = rbmsemisuplearn(rbm,x,ey,opts,chains_comb,chainsy_type )
%rbmsemisuplearn semisupervised learning function
% Combines unsupervised training objective with either 
% hybrid, discriminative or generative trainign using the formula:
%
%  L = L_type + L_unsup*opts.semisup_beta
%
%  for discription of semisupervised training objective see ref [2]
%   INPUTS:
%       rbm       : a rbm struct
%       x        : the initial state of the hidden units
%       ey        : one hot encoded labels if classRBM otherwise empty
%       opts      : opts struct. opts.train_type determines if CD or PCD should
%                   be used for generative training. otps.cdn determines the
%                   number of gibbs steps.
%                   opts.hybrid_alpha determines the weigthing of hybrid and
%                   generative training.
%  chains_comb    : PCD chains for otps.semisup_type and semisupervised train
%                   func
%  chainsy_type   : PCD chains for the tranining func det. by otps.semisup_type
%
%   OUTPUTS
%      A grads struct with the fields:
%       grads.dw   : w weights chainge normalized by minibatch size
%       grads.db   : bias of visible layer weight change norm by minibatch size
%                    (db is zero for the discriminative RBM)
%       grads.dc   : bias of hidden layer weight change norm by minibatch size
%       grads.du   : class label layer weight change norm by minibatch size
%       grads.dd   : class label hidden bias weight change norm by minibatch size
%       curr_err   : not used returns 0 
%  chains_comb     : updated PCD chains
%  chainsy_type    : updated PCD chains
%
%
%
% References
%    [1] H. Larochelle and Y. Bengio, Classification using discriminative
%        restricted Boltzmann machines,? ? 25th Int. Conf. Mach. ?, 2008.
%    [2] H. Larochelle and M. Mandel, ?Learning algorithms for the
%        classification restricted boltzmann machine,? J. Mach.  ?, 2012.
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
% %   See also RBMGENERATIVE, RBMDISCRIMINATIVE.

%split chains don't if it matters that i use the correct chains??
chains_semisup = chains_comb(1:opts.batchsize,:);
chains_type    = chains_comb(opts.batchsize+1:end,:);

rbm.classRBM = 0;
[g_semisup,~,chains_semisup,~] = rbmgenerative(rbm,opts.x_semisup_batch,[],...
    opts,chains_semisup,[]);


% combine the generative unsupervised training with either @rbmhybrid,
% @rbmgenerative or @rbmdiscriminative
rbm.classRBM = 1;
[g_type,~,chains_type,chainsy_type]= opts.semisup_type(rbm,x,ey,opts,...
    chains_type,chainsy_type);

chains_comb = [chains_semisup; chains_type];


weight_grads = @(type,semisup) type +  opts.semisup_beta * semisup;

grads.dw = weight_grads(g_type.dw,g_semisup.dw);
grads.dc = weight_grads(g_type.dc,g_semisup.dc);

% if the opts.semisup_type is
if isequal(opts.semisup_type,@rbmdiscriminative)
    % rbmdiscriminative outputs empty b
    grads.db = g_semisup.db;
else
    grads.db = weight_grads(g_type.db,g_semisup.db);
end
% these are not outputted from generative semisupervised training
grads.du = g_type.du;
grads.dd = g_type.dd;
curr_err = 0;
end






