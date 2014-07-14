function [ dw,db,dc,du,dd,curr_err,chains,chainsy ] = ...
    rbmdiscriminative( rbm,x,ey,opts,chains,chainsy )
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here


[n_hidden,n_visible]  = size(rbm.W);
n_classes = size(rbm.d,1);
n_samples = size(x,1);


% calculate probailities

% precalcualte activation of hidden units
cwx = bsxfun(@plus,rbm.W*x',rbm.c);

% loop over all classes and caluclate energies and probabilities
F = zeros(n_hidden,n_samples,n_classes);
class_log_prob = zeros(n_samples,n_classes);
for y = 1:n_classes
    F(:,:,y) = bsxfun(@plus,rbm.U(:,y),cwx);
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


%% OK hertil
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


dw = dw / opts.batchsize;
%db = db / opts.batchsize;
dc = dc / opts.batchsize;
dd = dd / opts.batchsize;
du = du / opts.batchsize;

curr_err = 0;
end

