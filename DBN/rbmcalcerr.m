function [ err ] = rbmcalcerr( rbm,x,y )
%RBMCALCERR Calculate the error in percent
if ~rbm.classRBM
    error('RBM is not a classification RBM')
end
pred_labels      = rbmpredict( rbm,x);
[~, true_labels] = max(y,[],2);
perf             = mean(pred_labels==true_labels);
err = 1-perf;
end

