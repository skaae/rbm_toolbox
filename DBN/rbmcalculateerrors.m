function [errors,errstr]= rbmcalculateerrors( rbm,errors,epoch,x_train,x_val,y_train,y_val,time)
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here
 n_samples = size(x_train,1);
     hlr    = num2str(rbm.learningrate(epoch));
    hmom   = num2str(rbm.momentum(epoch));
    
    [herr_train, ~] = accuracy(rbmclassprobs( rbm,x_train),y_train);
    if ~isempty(x_val)
    [herr_val, ~] = accuracy(rbmclassprobs( rbm,x_val),y_val);
    else
        herr_val = -Inf; 
    end
    
    errors.train(end+1) = herr_train;
    errors.val(end+1) = herr_val;
    errors.reconerror(end+1) = rbm.reconerror / n_samples;
    
    epochnr = ['Epoch ' num2str(epoch)];
    avg_err = [' Avg recon. err: ' num2str(  errors.reconerror(end)) '|'];
    lr_mom  = [' LR: ' hlr '. Mom.: ' hmom];
    htime   = ['  time: ',num2str(time)];
    perf    = [' TR: ' num2str(herr_train) ,' VAL: ', num2str(herr_val)];
    errstr = [epochnr avg_err lr_mom, htime perf];
    
   

end

