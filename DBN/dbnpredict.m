function [ predictions ] = dbnpredict(dbn,x)
%DBNPREDICT predict labels using classification DBN
%
%  INPUTS
%   dbn : A dbn struct
%   x   : matrix of samples  (n_samlples-by-n_features)
%
%  OUTPUT
%   predictions : [n_samples x 1] vector of predicted labels
%
% See also, DBNCLASSPROBS 
%
% Copyright Søren Sønderby July 2014
class_probs = dbnclassprobs( dbn,x);
predictions = predict(class_probs);
end
