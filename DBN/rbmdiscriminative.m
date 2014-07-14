function [ dw,db,dc,du,dd,curr_err,chains,chainsy ] = ...
    rbmdiscriminative( rbm,x,ey,opts,chains,chainsy )
%UNTITLED15 Summary of this function goes here
%   Detailed explanation goes here


[n_hidden,n_visible]  = size(rbm.W);
n_classes = size(rbm.d,1);
n_samples = size(x,1);

% init grads
dw = zeros(size(rbm.W));
du = zeros(size(rbm.U));
db = zeros(size(rbm.b));
dc = zeros(size(rbm.c));
dd = zeros(size(rbm.d));


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
class_normalizer = log_sum_exp_over_rows(class_log_prob); 
log_class_prob = bsxfun(@minus, class_log_prob,class_normalizer);
class_prob = exp(log_class_prob);


F_sigm = sigm(F);
F_sigm_prob = zeros(size(F_sigm));
for i = 1:n_classes
    F_sigm_prob(:,:,i)  = bsxfun(@times, F_sigm(:,:,i),class_prob(:,i)');
end

[~,class_labels] = max(ey,[],2);
for i = 1:n_classes
    
    %%  dw grad
    idx = find(i == class_labels);
    dw = dw +  F_sigm(:,idx,i)*x(idx,:) - F_sigm_prob(:,:,i)*x; 
    
    
    grad = sum(F_sigm(:,class_labels == i,i),2)- - sum(F_sigm_prob(:,:,i),2);
    %%  du grad
    du(:,i) = grad; 
    
    %% dc grad
    dc = dc + grad;
    
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

