function [ class_probs ] = rbmclassprobs( rbm,x)
%RBMCLASSPROBS calculate class probabilities for a classification RBM
% 
%  INPUTS
%   rbm       : A rbm struct
%          x  : matrix of samples  (n_samlples-by-n_features)
%   batchsize : optionally takes a minibatch size in which case the result
%               is calculated in minibatches to save memory
%
%  OUTPUT
%   class_prob_res : class probabilites for each class (n_samples-by-n_classes)
%
%  NOTES
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
% See also DBNCLASSPROBS
%
% Copyright(c) Søren Sønderby july 2014

% check if result should be calculated in batches 
n_samples = size(x,1);
n_classes = size(rbm.U,2);
if rem(n_samples,100) == 0
    chunksize = 250;
else
    % search for a chunks size, eventually we will hit 1
    chunksize = 200;
    while rem(n_samples,chunksize) ~= 0
        chunksize = chunksize - 1;
    end
end

chunks = chunkify(chunksize,x);


class_probs = zeros(n_samples,n_classes);
for i = 1:numel(chunks)
    samples = chunks{i}.start:chunks{i}.end;
    batch = x(samples,:);
    class_probs(samples,:) = rbmpygivenx(rbm,batch);
end



end








