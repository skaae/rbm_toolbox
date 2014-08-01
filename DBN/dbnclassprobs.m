function  class_probs = dbnclassprobs( dbn,x, batchsize )
%DBNCLASSPROBS calculates p(y|x) in a classification DBN
%
%  INPUTS
%   dbn       : A dbn struct
%   x         : matrix of samples  (n_samlples-by-n_features)
%   batchsize : optionally takes a minibatch size in which case the result
%               is calculated in minibatches to save memory
%
%  OUTPUT
%   class_probs : class probabilites for each class (n_samples-by-n_classes)
%
%  EXAMPLE
%   class_probs = dbnclassprobs( dbn,x )
%   pred        = predict(x)
%
% See also, DBNPREDICT

% Copyright Søren Sønderby july 2014

n_rbm = numel(dbn.rbm);



if ~dbn.rbm{n_rbm}.classRBM
    error('Class probabilities can only be calc. for classification DBN');
end
% pass data deterministicly from input to top RBM
for i = 1 : n_rbm-1
    x = rbm.rbmup(dbn.rbm{i},x,[],@sigm);
end

% at top RBM calculate class probabilities
if exist('batchsize','var')
    class_probs = rbmclassprobs( dbn.rbm{n_rbm},x,batchsize);
else
    class_probs = rbmclassprobs( dbn.rbm{n_rbm},x,batchsize);
end




end

