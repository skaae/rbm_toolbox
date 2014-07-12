function  class_probs = dbnclassprobs( dbn,x )
%DBNCLASSPROBS calculate class probabilities in a classification DBN
%  INPUTS
%   dbn : A dbn struct
%   x   : matrix of samples  (n_samlples-by-n_features)
%
%  OUTPUT
%   class_probs : class probabilites for each class (n_samples-by-n_classes)
%
% Copyright Søren Sønderby july 2014

n_rbm = numel(dbn.rbm);

if ~dbn.rbm{n_rbm}.classRBM
    error('Class probabilities can only be calc. for classification DBN');
end
% pass data deterministicly from input to top RBM
for i = 1 : n_rbm-1
    x = rbmup(dbn.rbm{i},x,[],@sigm);
end

% at top RBM calculate class probabilities
class_probs = rbmclassprobs( dbn.rbm{n_rbm},x);




end

