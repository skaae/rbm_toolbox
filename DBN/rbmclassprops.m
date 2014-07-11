function [ class_probs ] = rbmclassprops( rbm,x)
%RBMCLASSPROBS calculate class probabilities in a classification RBM
%  INPUTS
%   rbm : A rbm struct
%   x   : matrix of samples  (n_samlples-by-n_features)
%   
%  OUTPUT
%   class_probs : class probabilites for each class (n_samples-by-n_classes)
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
%  TODO
%    vectorize code?
%
% Copyright Søren Sønderby july 2014
n_visible = size(rbm.W,2);
if ~rbm.hintonDBN
    error('Class probabilities can only be calc. for classification RBM´s');
end
if size(x,2) ~= n_visible
    error('x has wrong dimensions');
end

n_hidden  = size(rbm.W,1);
n_classes = size(rbm.d,1);
n_samples = size(x,1);

% calculate probabilities for all samples
class_probs = zeros(n_samples,n_classes);
for t = 1:n_samples
    class_probs(t,:) = calcprobs(x(t,:));
end

    % function for calculating probabilities for single sample returns a vector
    % of size 1 x n_classes with class probabilities
    function probs = calcprobs(x_t)
        % precompute rbm.c(j)+rbm.W(j,:)*x; over all hidden units
        cwx = rbm.c + rbm.W * x_t';
  
        probs = zeros(1,n_classes);
        for y_idx = 1:n_classes
            freeenergy = -rbm.d(y_idx);
            for j = 1:n_hidden
                freeenergy = freeenergy - softplus(cwx(j)+rbm.U(j,y_idx));
                
            end
        probs(y_idx) = exp(-freeenergy);
        end
        probs = probs./sum(probs);
    end

end



