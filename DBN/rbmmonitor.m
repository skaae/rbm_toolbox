function [ perf,rbm ] = rbmmonitor(rbm,x,opts,val_samples,epoch)
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
if mod(epoch,opts.test_interval) == 0
    if rbm.classRBM
        train_probs = rbmclassprobs( rbm,x,opts.batchsize);
        %train_confusion = confusionmatrix(train_probs,opts.y_train);
        [train_err, train_om] = rbm.err_func(train_probs,opts.y_train);
        
        rbm.train_error(end+1) = train_err;
        rbm.train_error_measures{end+1} = train_om;
        if ~isempty(opts.x_val)
            val_probs = rbmclassprobs( rbm,opts.x_val,opts.batchsize);
            %val_confusion = confusionmatrix(val_probs,opts.y_val);
            [val_err, val_om] = rbm.err_func(val_probs,opts.y_val);
            
            rbm.val_error(end+1) = val_err;
            rbm.val_error_measures{end+1} = val_om;
            val_err = num2str(rbm.val_error(end));
        else
            val_err = 'NA';
        end
        
        perf = sprintf('  | Tr: %5f - Val: %s' ,...
            rbm.train_error(end),val_err);
    else
        %         warning('Remimpelent rbmfreeenergy');
        %         e_val = rbmfreeenergy(rbm,opts.x_val,opts.y_val);
        %         e_train = rbmfreeenergy(rbm,x(val_samples,:),...
        %             opts.y_train(val_samples,:));
        %         rbm.energy_ratio(end+1) = e_val / e_train;
        %         perf = sprintf(' | Energy ratio %f5', e_val / e_train);
        perf = '';
    end
else  % not a sampling epoch 
    perf = '.';
end

end

