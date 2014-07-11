function [ predictions ] = dbnpredict(dbn,x)
%DBNPREDICT predict labels using hintonDBN
%  INPUTS
%   dbn : A dbn struct
%   x   : matrix of samples  (n_samlples-by-n_features)
%   
%  OUTPUT
%   predictions : [n_samples x 1] vector of predicted labels
%
% Copyright Søren Sønderby July 2014
class_probs = dbnclassprobs( dbn,x);
[~, predictions] = max(class_probs,[],2);
end
