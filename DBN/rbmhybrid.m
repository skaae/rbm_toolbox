function [ grads,curr_err,chains,chainsy ] = rbmhybrid(rbm,x,ey,opts,chains,chainsy,debug)
%RBMDISCRIMINATIVE calcualte weight updates for hybrid RBM
%  for discription of hybrid training objective see ref [1,2]
%  refer to RBMGENERATIVE and RBMDISCRIMINATIVE for description of the
%  training objectives.
%
%   INPUTS:
%       rbm       : a rbm struct
%       x         : the initial state of the hidden units
%       ey        : one hot encoded labels if classRBM otherwise empty
%       opts      : opts struct. opts.train_type determines if CD or PCD should
%                   be used for generative training. otps.cdn determines the
%                   number of gibbs steps.
%                   opts.hybrid_alpha determines the weigthing of hybrid and
%                   generative training.
%       chains    : not used, pass in anything
%       chainsy   : not used, pass in anything
%       debug     : if it exists and is 1 save intermediate values to file
%                   currentworkdir/test_rbmhybrid.mat
%
%   OUTPUTS
%      A grads struct with the fields:
%       grads.dw   : w weights chainge normalized by minibatch size
%       grads.db   : bias of visible layer weight change norm by minibatch size
%                    (db is zero for the discriminative RBM)
%       grads.dc   : bias of hidden layer weight change norm by minibatch size
%       grads.du   : class label layer weight change norm by minibatch size
%       grads.dd   : class label hidden bias weight change norm by minibatch size
%       curr_err   : current squared error normalized by minibatch size
%       chains     : updated value of chains for visible units
%       chainsy    : updated value of chains for label visible units.
%
% REFERENCES
%    [1] H. Larochelle and Y. Bengio, ?Classification using discriminative
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
%  See also RBMGENERATIVE, RBMDISCRIMINATIVE RBMSEMISUPLEARN
%
% Copyright Søren Sønderby June 2014



[grads_gen,curr_err,chains,chainsy] = rbmgenerative(rbm,x,ey,opts,chains,chainsy);

[grads_dis,~,~,~]= rbmdiscriminative(rbm,x,ey,opts,[],[]);


if exist('debug','var') && debug == 1
    warning('Debugging rbmhybrid')
    save('test_rbmhybrid','grads_gen','grads_dis');
end

grads = struct();
grads.dw = grads_dis.dw + opts.hybrid_alpha * grads_gen.dw;
grads.db = grads_dis.db + opts.hybrid_alpha * grads_gen.db;
grads.dc = grads_dis.dc + opts.hybrid_alpha * grads_gen.dc;
grads.du = grads_dis.du + opts.hybrid_alpha * grads_gen.du;
grads.dd = grads_dis.dd + opts.hybrid_alpha * grads_gen.dd;

end






