function [ predictions ] = rbmpredict(rbm,x)
%RBMPREDICT predict labels using classRBM
%  INPUTS
%   rbm : A rbm struct
%   x   : matrix of samples  (n_samlples-by-n_features)
%
%  OUTPUT
%   predictions : [n_samples x 1] vector of predicted labels
%
% Copyright Søren Sønderby July 2014
class_probs = rbmclassprobs( rbm,x);
[~, predictions] = max(class_probs,[],2);
end

