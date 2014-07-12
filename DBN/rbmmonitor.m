function [ perf,rbm ] = rbmmonitor(rbm,x,opts,val_samples,epoch)
%RBMMONITOR calculate perforamnce
%   If the RBM is a classRBM the function augments rbm.train_perf and 
%   rbm.val_perf with the newest accuracies. 
%   If generativeRBM calcualte the free energy ratio and update rbm.energy_ratio
%
%   val_samples is a vector of the same length as the number of  validation
%   samples. Used when free energies are compared.
%
%   Returns a string for dispaly
% Copyright Søren Sønderby July 2014
if mod(epoch,opts.test_interval) == 0 && ~isempty(opts.x_val)
    % a) lassRBM calculate validation and training error
    % b) generativeRBM  calculate ratio of free energy fe_val /fe_train, if
    %    much lower than 1 we are overfitting
    if rbm.classRBM
        rbm.train_perf(end+1) = rbmcalcerr(rbm,x,opts.y_train);
        rbm.val_perf(end+1) = rbmcalcerr(rbm,opts.x_val,opts.y_val);
        perf = sprintf('  | Tr: %5f - Val: %5f' ,...
            rbm.train_perf(end),rbm.val_perf(end));
    else
        warning('Remimpelent rbmfreeenergy');
        e_val = rbmfreeenergy(rbm,opts.x_val,opts.y_val);
        e_train = rbmfreeenergy(rbm,x(val_samples,:),...
            opts.y_train(val_samples,:));
        rbm.energy_ratio(end+1) = e_val / e_train;
        perf = sprintf(' | Energy ratio %f5', e_val / e_train);
    end
else
    perf = '.';
end

end

