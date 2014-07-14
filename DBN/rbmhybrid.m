function [ grads,curr_err,chains,chainsy ] = rbmhybrid(rbm,x,ey,opts,chains,chainsy )
%RBMDISCRIMINATIVE calcualte weight updates for hybrid RBM
%  for discription of hybrid training objective see ref [1,2]
%  refer to RBMGENERATIVE and RBMDISCRIMINATIVE for description of the
%  training objectives.
%
%
%   INPUTS:
%       rbm       : a rbm struct
%       v0        : the initial state of the hidden units
%       ey        : one hot encoded labels if classRBM otherwise empty
%       opts      : opts struct. opts.train_type determines if CD or PCD should
%                   be used for generative training. otps.cdn determines the
%                   number of gibbs steps.
%                   opts.hybrid_alpha determines the weigthing of hybrid and
%                   generative training.
%       chains    : not used, pass in anything
%       chainsy   : not used, pass in anything
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
%
%
% References
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
% Copyright Søren Sønderby June 2014
% %   See also RBMGENERATIVE, RBMDISCRIMINATIVE.

rbm.classRBM = 0;
rbm_gen = @() rbmgenerative(rbm,x,ey,opts,chains,chainsy);

rbm.classRBM = 1;
rbm_dis = @() rbmdiscriminative(rbm,x,ey,opts,chains,chainsy);


results = {};
for i = 1:2
    if i == 1  % generative call
        [g,err,chainsupd,chainsyupd] = rbm_gen();
        results{i}.type = 'generative';
        fprintf('.')
    else
        [g,err,chainsupd,chainsyupd]=rbm_dis();
        results{i}.type = 'discriminative';
        fprintf('.')
    end
    
    
    results{i}.grads = g;
    results{i}.c_err = err;
    results{i}.chains= chainsupd;
    results{i}.chainsy = chainsyupd;
end

weight_grads = @(gen,dis) (1-opts.hybrid_alpha)*dis +  opts.hybrid_alpha*dis;

gen = results{1};
dis = results{2};
grads.dw = weight_grads(gen.grads.dw,dis.grads.dw);
grads.db = weight_grads(gen.grads.db,dis.grads.db);
grads.dc = weight_grads(gen.grads.dc,dis.grads.dc);
grads.du = weight_grads(gen.grads.du,dis.grads.du);
grads.dd = weight_grads(gen.grads.dd,dis.grads.dd);

chains = gen.chains;
chainsy = gen.chainsy;
curr_err = gen.c_err;


end






