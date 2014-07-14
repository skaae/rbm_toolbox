function [ grads,curr_err,chains,chainsy ] = ...
                               rbmdiscriminative( rbm,x,ey,opts,chains,chainsy )
%RBMDISCRIMINATIVE calcualte weight updates for discriminative RBM
%
%
%   INPUTS:
%       rbm       : a rbm struct
%       v0        : the initial state of the hidden units
%       ey        : one hot encoded labels if classRBM otherwise empty
%       opts      : opts struct
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
%       curr_err   : not used, returns 0
%       chains     : not used, returns []
%       chainsy    : not used, returns []
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


[n_hidden,n_visible]  = size(rbm.W);
n_classes = size(rbm.d,1);
n_samples = size(x,1);


% calculate probailities

% precalcualte activation of hidden units
cwx = bsxfun(@plus,rbm.W*x',rbm.c);

% loop over all classes and caluclate energies and probabilities
%F = zeros(n_hidden,n_samples,n_classes);
 F = bsxfun(@plus, permute(rbm.U, [1 3 2]), cwx);
class_log_prob = zeros(n_samples,n_classes);
for y = 1:n_classes
    %F(:,:,y) = bsxfun(@plus,rbm.U(:,y),cwx);
    class_log_prob(:,y) =  sum( softplus(F(:,:,y)), 1)+ rbm.d(y);
end

%normalize probabilities in numerically stable way
class_prob = exp(bsxfun(@minus, class_log_prob, max(class_log_prob, [], 2)));
class_prob = bsxfun(@rdivide, class_prob, sum(class_prob, 2));



F_sigm = sigm(F);
F_sigm_prob = zeros(size(F_sigm));
for i = 1:n_classes
    F_sigm_prob(:,:,i)  = bsxfun(@times, F_sigm(:,:,i),class_prob(:,i)');
end

% init grads
dw = zeros(size(rbm.W));
du = zeros(size(rbm.U));
db = zeros(size(rbm.b));
dc = zeros(size(rbm.c));
dd = zeros(size(rbm.d));


[~,class_labels] = max(ey,[],2);
temp = zeros(500,10);
for i = 1:n_classes
    
    %%  dw grad
    idx = find(i == class_labels);
    dw = dw +  F_sigm(:,idx,i)*x(idx,:) - F_sigm_prob(:,:,i)*x;
    
    
    %%  du grad
    du(:,i) = sum(F_sigm(:,class_labels == i,i),2) - sum(F_sigm_prob(:,:,i),2);
    
    
    
    %% dc grad
    dc = dc + sum(F_sigm(:,class_labels == i,i),2) - sum(F_sigm_prob(:,:,i),2);
    
end

%% dd grad
dd = sum(ey - class_prob,1)';


grads.dw = dw / opts.batchsize;
grads.db = db / opts.batchsize;
grads.dc = dc / opts.batchsize;
grads.dd = dd / opts.batchsize;
grads.du = du / opts.batchsize;

curr_err = 0;
end

