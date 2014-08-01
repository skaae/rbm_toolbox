function [best_err,patience,best_str,best_rbm] = rbmearlystopping(rbm,opts,best_err,patience)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

if rbm.early_stopping &&  mod(epoch,opts.test_interval) == 0
    isbest = 0;
    % for classification RBM's check if validatio is better than
    % current best
    if  rbm.classRBM ==1 && best_err > rbm.val_error(end)
        isbest = 1;
        
        % for generative RBM's check if the ratio is below 0
    elseif rbm.classRBM ==0 && best_err > rbm.energy_ratio(end) 
        if  rbm.energy_ratio(end) > 0.99 % check for overfitting
        isbest = 1;
        end
    end
    
    if isbest
        best_str = ' ***';
        best_err = rbm.val_error(end);
        best_rbm = rbm;
        patience = rbm.patience;
        
    else
        best_str = '';
        patience = patience-1;
    end
end

