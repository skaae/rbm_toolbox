function rbm = rbmtrain(rbm, x_train,opts)
%%RBMTRAIN trains a single RBM
% notation:
%    w  : weights
%    b  : bias of visible layer
%    c  : bias of hidden layer
%  Modified by Søren Sønderby June 2014

% SETUP and checking
assert(isfloat(x_train), 'x must be a float');
assert(all(x_train(:)>=0) && all(x_train(:)<=1), 'all data in x must be in [0:1]');
m = size(x_train, 1);
numbatches = m / opts.batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches not integer');

% use validation set or not
use_val = ifelse(~isempty(opts,'x_val'),1,0);




% RUN epochs
init_chains = 1;
chains = [];
for i = 1 : opts.numepochs
    kk = randperm(m);
    err = 0;
    for l = 1 : numbatches
        v0 = x_train(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
        
        if strcmp(opts.traintype,'PCD') && init_chains == 1
            % init chains in first epoch if Persistent contrastive divergence
            chains = v0;
            init_chains = 0;
        end
        
        % Collect rbm statistics with CD or PCD
        [dw,db,dc,c_err,chains] = rbmstatistics(rbm,v0,opts,opts.traintype,chains);
        
        %update weights, LR,decay and momentum 
        rbm = rbmapplygrads(rbm,dw,db,dc,v0,i);
        err = err + c_err;
    end
    

    
    % if the training data energy is much lower than the validation energy
    % rasie a overfitting warning. (i.e the ratio becomes <1)
    if mod(i,5) == 0 && use_val == 1
        e_val = rbmenergy(rbm,opts.x_val);
        e_train = rbmenergy(rbm,x_train(1:size(opts.x_val,1),:));
        ratio = e_val / e_train;
        oft = ifelse(ratio<0.8,'(overfitting)','(OK)');
        energy = sprintf('. E_Val / E_train %4.3f %s',ratio,oft);
    else
        energy = '.';
    end
    
    % display output
    epochnr = ['Epoch ' num2str(i) '/' num2str(opts.numepochs) '.'];
    avg_err = [' Avg recon. err: ' num2str(err / numbatches) '|'];
    lr_mom  = [' LR: ' num2str(rbm.curLR) '. Mom.: ' num2str(rbm.curMomentum)];
    disp([epochnr avg_err lr_mom energy]);
    
    
end
end
