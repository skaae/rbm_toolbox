function hrbm = rbmtraingpu(hrbm, x_train,hopts)
%RBMTRAIN trains a single RBM
%
% NOTATION:
% data  : all data given as      [n_samples   x #vis]
%    W  : vis - hid weights      [ #hid       x #vis ]
%    U  : label - hid weights    [ #hid       x #n_classes ]
%    b  : bias of visible layer  [ #vis       x 1]
%    c  : bias of hidden layer   [ #hid       x 1]
%    d  : bias of label layer    [ #n_classes x 1]
%
% See also DBNTRAIN
%
% Copyright Søren Sønderby June 2014

% SETUP and checking
assert(isfloat(x_train), 'x must be a float');
assert(all(x_train(:)>=0) && all(x_train(:)<=1), 'all data in x must be in [0:1]');
n_samples = size(x_train, 1);
[n_hidden, ~] = size(hrbm.W);

numbatches = n_samples / hopts.batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches not integer');


%%% show gpu info
gpu = gpuDevice();
reset(gpu);
wait(gpu);
disp(['GPU memory available (Gb): ', num2str(gpu.FreeMemory / 10^9)]);

% hrbm is HOSTrbm which is not calculated on gpu.
hrbm.gpu = 0;


% use validation set or not in calculation of free energy
if ~isempty(hopts.x_val)
    n_val_samples   = size(hopts.x_val,1);
    samples         = randperm(size(x_train,1));
    % if size of val set is larger than train set use trainset size otherwise
    % use size of validation set
    size_val_sample = ifelse(n_samples>=n_val_samples, n_val_samples, n_samples);
    x_samples     = samples(1:size_val_sample);
    val_samples       = 1:size_val_sample;
else
    val_samples = [];
end

earlystop.best_err = Inf;
earlystop.patience  = hrbm.patience;


% RUN epochs
init_chains = 1;
chains = [];
chainsy = [];
best_str = '';


if isequal(hrbm.train_func,@rbmsemisuplearn)
    semisup = 1;
    l_semisup = 0;
    n_samples_semisup = size(hopts.x_semisup,1);
    numbatches_semisup = n_samples_semisup / hopts.batchsize;
    assert(rem(numbatches_semisup, 1) == 0, 'semisup numbatches not integer');
else
    semisup = 0;
end


drbm = cpRBMtoGPU(hrbm);

for epoch = 1 : hopts.numepochs
    kk = randperm(n_samples);
    
    if semisup
        kk_semisup = randperm(n_samples_semisup);
    end
    
    err = 0;
    for l = 1 : numbatches
        v0 = extractminibatch(kk,l,hopts.batchsize,x_train,opts);
        if drbm.classRBM == 1
            ey = extractminibatch(kk,l,hopts.batchsize,hopts.y_train,opts);
        else
            ey = [];
        end
        
        % create batches for semisupervised leanring
        if semisup == 1
            l_semisup = l_semisup + 1;
            if l_semisup > numbatches_semisup
                l_semisup = 1;
            end
            hopts.x_semisup_batch = extractminibatch(kk_semisup,...
                numbatches_semisup,hopts.batchsize,hopts.x_semisup,opts);
        end
        
        
        if strcmp(hopts.traintype,'PCD') && init_chains == 1
            % init chains in first epoch if Persistent contrastive divergence
            
            % augment semisup PCD chains starting position
            if semisup
                
                % init semisup chains at mean training set values
                % not sure if that is correct?
                meany  = samplematrix(repmat(mean(hopts.y_train,1),hopts.batchsize,1));
                chains = [hopts.x_semisup_batch; v0;];
                chainsy = [meany;ey;];
            else
                chains = v0;
                chainsy = ey;
            end
            init_chains = 0;
        end
        
        if drbm.dropout_hidden > 0
            drbm.hidden_mask = (rbm.rand(size(n_hidden,hopts.batchsize)) > drbm.dropout_hidden);
        end
        
        % calculate rbm gradients
        [grads,c_err,chains,chainsy]= drbm.train_func(drbm,v0,ey,hopts,chains,chainsy);
        
        err = err + c_err;
        %fprintf('%d\n',c_err)
        
        %update weights, LR,decay and momentum
        drbm = rbmapplygrads(drbm,grads,v0,ey,epoch);
    end
    drbm.error(end+1) = err / numbatches;
    
    % calc train/val performance.
    [perf,drbm] = rbmmonitor(drbm,x_train,hopts,x_samples,val_samples,epoch);
    earlystop  = rbmearlystopping(drbm,hopts,earlystop,epoch);
    
    % stop training?
    if drbm.early_stopping && earlystop.patience < 0
        disp('No more Patience. Return best RBM')
        earlystop.best_rbm.val_error = drbm.val_error;
        earlystop.best_rbm.train_error = drbm.train_error;
        earlystop.best_rbm.error = drbm.error;
        drbm = earlystop.best_rbm;
        
        break;
    end
% display output
epochnr = ['Epoch ' num2str(epoch) '/' num2str(hopts.numepochs) '.'];
avg_err = [' Avg recon. err: ' num2str(err / numbatches) '|'];
lr_mom  = [' LR: ' num2str(drbm.curLR) '. Mom.: ' num2str(drbm.curMomentum)];
disp([epochnr avg_err lr_mom perf earlystop.best_str]);    
end

hrbm = cpRBMtoHost(drbm);


end




