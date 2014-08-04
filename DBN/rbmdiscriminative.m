function [grads,curr_err,chains,chainsy] = rbmdiscriminative(rbm,x,ey,opts,chains,chainsy,debug)
%RBMDISCRIMINATIVE calculate weight updates for discriminative RBM
%
%   INPUTS:
%       rbm       : a rbm struct
%       x        : the initial state of the hidden units
%       ey        : one hot encoded labels if classRBM otherwise empty
%       opts      : opts struct
%       chains    : not used, pass in anything
%       chainsy   : not used, pass in anything
%       debug     : if it exists and is 1 save intermediate values to file
%                   currentworkdir/test_rbmdiscriminative.mat
%
%   OUTPUTS
%      A grads struct with the fields:
%       grads.dw   : w weights chainge normalized by minibatch size
%       grads.db   : bias of visible layer weight change norm by minibatch size
%                    (db is zero for the discriminative RBM, returns [])
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
% See also RBMGENERATIVE RBMHYBRID RBMSEMISUPLEARN
%
% Copyright Søren Sønderby June 2014
n_classes = size(rbm.d,1);

[p_y_given_x, F]= rbmpygivenx(rbm,x,'train');

F_sigm = sigm(F);
F_sigm_prob = rbm.zeros(size(F_sigm));
for c = 1:n_classes
    F_sigm_prob(:,:,c)  = bsxfun(@times, F_sigm(:,:,c),p_y_given_x(:,c)');
end

% init grads
dw = rbm.zeros(size(rbm.W));
du = rbm.zeros(size(rbm.U));
dc = rbm.zeros(size(rbm.c));

class_labels = predict(ey);
%b = bs,F_sigm_prob,permute(repmat(x,1,1,n_classes),[3,1,2]));


%       a=arrayfun(@(c) test(c,F_sigm,class_labels,x),...
%                      1:n_classes,'uni',false);
%       b = arrayfun(@(c) F_sigm_prob(:,:,c)*x,...
%           1:n_classes,'uni',false);
%
%
%     for c = 1:n_classes
%         dw = dw + a{c} - b{c};
%     end
%
%     function a = test(c,F_sigm,class_labels,x)
%         % find idx for current class
%      l_idx = find(c == class_labels);
%      a = F_sigm(:,l_idx,c)*x(l_idx,:);
%
%     end
%


% f1 = size(F_sigm);
% x1 = size(x);
% a = reshape(sum(bsxfun(@times,reshape(F_sigm,[f1(1),1,f1(2:3)]),...
%                       reshape(x',1,x1(2),[])),3),f1(1),x1(2),[]);

for c = 1:n_classes
    %  dw grad
    
    % find idx for current class
    bin_idx =  class_labels == c;
    
    lin_idx = find(c == class_labels);
    
    a = F_sigm(:,lin_idx,c)*x(lin_idx,:);
    b = F_sigm_prob(:,:,c)*x;
    dw = dw + a-b;
    
    %  du grad
    du(:,c) = sum(F_sigm(:,bin_idx,c),2) - sum(F_sigm_prob(:,:,c),2);
    
    % dc grad
    dc = dc + sum(F_sigm(:,bin_idx,c),2) - sum(F_sigm_prob(:,:,c),2);
    
end

% dd grad
dd = sum(ey - p_y_given_x,1)';

% debugging
if exist('debug','var') && debug == 1
    warning('Debugging rbmdiscriminative')
    save('test_rbmdiscriminative','p_y_given_x','dd');
end

% create grads struct and scale grad by minibatch size.
grads.dw = dw / opts.batchsize;
grads.db = zeros(size(rbm.b));
grads.dc = dc / opts.batchsize;
grads.dd = dd / opts.batchsize;
grads.du = du / opts.batchsize;

curr_err = 0;
chains   = [];
chainsy  = [];
end

