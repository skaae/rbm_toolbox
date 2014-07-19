function [ grads,curr_err,chains_comb,chainsy_comb ] = rbmsemisuplearn(rbm,x,ey,opts,chains_comb,chainsy_comb )
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
%    See also RBMGENERATIVE, RBMDISCRIMINATIVE.

%split chains don't if it matters that i use the correct chains??
if strcmp(opts.traintype,'PCD')
    chains_semisup  = chains_comb(1:opts.batchsize,:);
    chains_type     = chains_comb(opts.batchsize+1:end,:);
    chainsy_semisup = chainsy_comb(1:opts.batchsize,:);
    chainsy_type     = chainsy_comb(opts.batchsize+1:end,:);
else
    chains_semisup = []; chains_type = [];
    chainsy_semisup = []; chainsy_type = [];
end


%sample p(y|x)
[p_y_given_x, ~]  = rbmpyx( rbm,x,'train');

[g_semisup,~,chains_semisup,chainsy_semisup] = rbmgenerative(rbm,...
         opts.x_semisup_batch, p_y_given_x,opts,chains_semisup,chainsy_semisup);


% combine the generative unsupervised training with either @rbmhybrid,
% @rbmgenerative or @rbmdiscriminative
[g_type,~,chains_type,chainsy_type]= opts.semisup_type(rbm,x,ey,opts,...
    chains_type,chainsy_type);

chains_comb = [chains_semisup; chains_type];
chainsy_comb = [chainsy_semisup; chainsy_type];


weight_grads = @(type,semisup) type +  opts.semisup_beta * semisup;

grads.dw = weight_grads(g_type.dw,g_semisup.dw);
grads.db = weight_grads(g_type.db,g_semisup.db);
grads.dc = weight_grads(g_type.dc,g_semisup.dc);
grads.du = weight_grads(g_type.du,g_semisup.du);
grads.dd = weight_grads(g_type.dd,g_semisup.dd);

curr_err = 0;


  

end






