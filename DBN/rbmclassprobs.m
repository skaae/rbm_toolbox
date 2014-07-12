function [ class_prob ] = rbmclassprobs( rbm,x)
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


%pre calculate
cwx = repmat(rbm.c,1,n_samples)+rbm.W*x';
F = zeros(n_samples,n_classes);
for y = 1:n_classes
    rep_U = repmat(rbm.U(:,y),1,n_samples);
    oyj = cwx + rep_U;
    F(:,y) =  sum( softplus(oyj ) )+ rbm.d(y);
end
class_normalizer = log_sum_exp_over_rows(F'); 
log_class_prob = F - repmat(class_normalizer',1,n_classes); 
class_prob = exp(log_class_prob);

end
%
%the top implementation seems to be slighly faster

% for t = 1:n_samples
%     freeenergy = -sum( softplus(repmat(cwx(:,t),1,n_classes) +  rbm.U )) - rbm.d';
%     probs = exp(-freeenergy);
%     class_probs(t,:) = probs ./ sum(probs);
% end

% %%% FORDEBUGGING
% class_probsb = zeros(n_samples,n_classes);
% cprobs = zeros(n_samples,n_classes);
% log_class_prob = zeros(n_samples,n_classes);
% class_normalizer = [];
% for t = 1:n_samples
%     if t == 50
%         a = 1;
%     end
%     [p cp lp cn] = calcprobs(x(t,:));
%     class_probsb(t,:) = p;
%     cprobs(t,:) = cp;
%     log_class_prob(t,:) = lp;
%     class_normalizer(end+1) = cn;
% end
%
% a = 1;
%     % function for calculating probabilities for single sample returns a vector
%     % of size 1 x n_classes with class probabilities
% function [class_prob cprobs log_class_prob class_normalizer] = calcprobs(x_t)
% % precompute rbm.c(j)+rbm.W(j,:)*x; over all hidden units
%         cwx = rbm.c + rbm.W * x_t';
%
%         cprobs = zeros(1,n_classes);
%         for y_idx = 1:n_classes
%             freeenergy = -rbm.d(y_idx);
%             for j = 1:n_hidden
%                 freeenergy = freeenergy - softplus(cwx(j)+rbm.U(j,y_idx));
%
%             end
%         cprobs(y_idx) = -freeenergy;
%         end
%         class_normalizer = log_sum_exp_over_rows(cprobs'); % log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities. size: <1> by <number of data cases>
%         log_class_prob = cprobs - class_normalizer; % log of probability of each class. size: <number of classes, i.e. 10> by <number of data cases>
%         class_prob = exp(log_class_prob); % probability of each class. Each column (i.e. each case) sums to 1. size: <number of classes, i.e. 10> by <number of data cases>
%
%
%
%
% end







