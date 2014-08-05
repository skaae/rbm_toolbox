function [ perf,rbm ] = rbmmonitor(rbm,x,opts,x_samples,val_samples,epoch)
%RBMMONITOR monitor rbm performance
% Internal function used by RBMTRAIN
% If the RBM is a classRBM the function augments rbm.train_perf and
% rbm.val_perf with the newest accuracies.
% If generativeRBM calcualte the free energy ratio and update rbm.energy_ratio
%
% val_samples is a vector of the same length as the number of  validation
% samples. Used when free energies are compared.
%
% INPUTS:
%         rbm : a rbm struct
%           x : the initial state of the hidden units
%        opts : opts struct
% val_samples : sample numbers in validatio set to be used for calculation
%               of free energies.
%     epoch   : current epoch number
%
% Copyright Søren Sønderby July 2014
perf = '.';

if mod(epoch,opts.test_interval) == 0
    if rbm.classRBM
        train_probs = rbmclassprobs( rbm,x,opts.batchsize);
        [train_err, train_om] = rbm.err_func(train_probs,opts.y_train);
        rbm.train_error(end+1) = train_err;
        rbm.train_error_measures{end+1} = train_om;
        
        if ~isempty(opts.x_val)
            val_probs = rbmclassprobs( rbm,opts.x_val,opts.batchsize);
            [val_err, val_om] = rbm.err_func(val_probs,r.y_val);
            
            rbm.val_error(end+1) = val_err;
            rbm.val_error_measures{end+1} = val_om;
            val_err = num2str(rbm.val_error(end));
        else
            val_err = 'NA';
        end
        
        perf = sprintf('  | Tr: %5f - Val: %s' ,...
            rbm.train_error(end),val_err);
    
    % non class RBM calculate free energy ratio
    elseif ~isempty(opts.x_val)
        x_s = x(x_samples,:);
        x_val_s = opts.x_val(val_samples,:);
        rbm.energy_ratio(end+1) = rbmfreeenergyratio(rbm,x_s,x_val_s);
        perf = sprintf(' | Energy ratio %f5', rbm.energy_ratio(end));
    end
end

end

