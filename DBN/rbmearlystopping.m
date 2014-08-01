function earlystop = rbmearlystopping(rbm,opts,earlystop,epoch)
%RBMEARLYSTOP applies early stopping if enabled
%   Internal function.
%   If earlystopping is enabled the function tests wether the 
%   current performance is better than the best performance seen.
%   For classification RBM the validatio error is checked for non classification
%   RBM's the ratio of free energies is checked. 


if rbm.early_stopping &&  mod(epoch,opts.test_interval) == 0
    isbest = 0;
    % for classification RBM's check if validatio is better than
    % current best
    if  rbm.classRBM ==1 && earlystop.best_err > rbm.val_error(end)
        isbest = 1;
        err = rbm.val_error(end);
        
        % for generative RBM's check if the ratio is below 0
    elseif rbm.classRBM ==0 && earlystop.best_err > rbm.energy_ratio(end) 
        if  rbm.energy_ratio(end) > 0.99 % check for overfitting
        isbest = 1;
        err = rbm.energy_ratio(end);
        end
    end
    
    if isbest
        earlystop.best_str = ' ***';
        earlystop.best_err = err;
        earlystop.best_rbm = rbm;
        earlystop.patience = rbm.patience;
        
    else
        earlystop.best_str = '';
        earlystop.patience = earlystop.patience-1;
    end
end

