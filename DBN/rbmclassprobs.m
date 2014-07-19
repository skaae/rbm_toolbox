function [ class_prob_res ] = rbmclassprobs( rbm,x,batchsize)
%RBMCLASSPROBS calculate class probabilities in a classification RBM
%  INPUTS
%   rbm : A rbm struct
%   x   : matrix of samples  (n_samlples-by-n_features)
%   batchsize: optionally takes a minibatch size in which case the result
%              is calculated in minibatches to save memory
%
%  OUTPUT
%   class_prob_res : class probabilites for each class (n_samples-by-n_classes)
%
%  NOTES SEE
%   see equation 2 of the paper:
%   Learning algorithms for the classification restricted boltzmann machine
%
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
% Copyright Søren Sønderby july 2014
n_visible = size(rbm.W,2);
if ~rbm.classRBM
    error('Class probabilities can only be calc. for classification RBM´s');
end
if size(x,2) ~= n_visible
    error('x has wrong dimensions');
end

n_hidden  = size(rbm.W,1);
n_classes = size(rbm.d,1);
n_samples = size(x,1);



if nargin == 3
    numbatches = n_samples / batchsize;
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    chunks = chunkify(batchsize,x);
    
else
    chunks = chunkify(n_samples,x);
end

class_prob_res = [];
for i = 1:numel(chunks)
    minibatch = x(chunks{i}.start:chunks{i}.end,:);
    [class_prob, ~] = rbmpyx(rbm,minibatch,'test');
    class_prob_res = [class_prob_res;class_prob];
end



end








