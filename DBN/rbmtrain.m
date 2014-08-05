function rbm = rbmtrain(rbm, x_train,opts)
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
[n_hidden, n_visible] = size(rbm.W);

numbatches = n_samples / opts.batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches not integer');

% use validation set or not in calculation of free energy
if ~isempty(opts.x_val)
    n_val_samples   = size(opts.x_val,1);
    samples         = randperm(size(x_train,1));
    % if size of val set is larger than train set use trainset size otherwise
    % use size of validation set
    size_val_sample = ifelse(n_samples>=n_val_samples, n_val_samples, n_samples);
    x_samples     = samples(1:size_val_sample);
    val_samples       = 1:size_val_sample;
else
    val_samples = [];
    x_samples    = [];
end

earlystop.best_err = Inf;
earlystop.patience  = rbm.patience;
earlystop.best_str = '';


% RUN epochs
init_chains = 1;
chains = [];
chainsy = [];
best_str = '';

% %% calculate normalization and normalize for x train, x val and x semisup
% rbm.xt_MU = mean(x_train,1);
% x_train = bsxfun(@minus,x_train,rbm.xt_MU);
% 
% if rbm.classRBM
%     rbm.yt_MU = mean(opts.y_train,1);
%     opts.y_train = bsxfun(@minus,opts.y_train,rbm.yt_MU);
% end
% if ~isempty(opts.x_val)
%     rbm.xv_MU = mean(opts.x_val);
%     rbm.yv_MU = mean(opts.y_val);
%     opts.y_val =  bsxfun(@minus,opts.y_val,rbm.yv_MU);
%     opts.x_val =  bsxfun(@minus,opts.x_val,rbm.xv_MU);
% 
% end
% if ~isempty(opts.x_semisup)
%     rbm.xs_MU = mean(opts.x_semisup);
%     opts.x_semisup =  bsxfun(@minus,opts.x_semisup,opts.xs_MU);
% end


if isequal(rbm.train_func,@rbmsemisuplearn)
    semisup = 1;
    l_semisup = 0;
    n_samples_semisup = size(opts.x_semisup,1);
    numbatches_semisup = n_samples_semisup / opts.batchsize;
    assert(rem(numbatches_semisup, 1) == 0, 'semisup numbatches not integer');
else
    semisup = 0;
end

for epoch = 1 : opts.numepochs
    kk = randperm(n_samples);
    
    if semisup
        kk_semisup = randperm(n_samples_semisup);
    end
    
    err = 0;
    
    % in each epoch update rbm parameters
    rbm.curMomentum     = rbm.momentum(epoch);
    rbm.curLR           = rbm.learningrate(epoch,rbm.curMomentum);
    rbm.curCDn          = rbm.cdn(epoch);
    
    
    for l = 1 : numbatches
        v0 = extractminibatch(kk,l,opts.batchsize,x_train,opts);
        if rbm.classRBM == 1
            ey = extractminibatch(kk,l,opts.batchsize,opts.y_train,opts);
        else
            ey = [];
        end
        
        % create batches for semisupervised leanring
        if semisup == 1
            l_semisup = l_semisup + 1;
            if l_semisup > numbatches_semisup
                l_semisup = 1;
            end
            opts.x_semisup_batch = extractminibatch(kk_semisup,...
                numbatches_semisup,opts.batchsize,opts.x_semisup,opts);
        end
        
        
        if strcmp(opts.traintype,'PCD') && init_chains == 1
            % init chains in first epoch if Persistent contrastive divergence
            
            % augment semisup PCD chains starting position
            if semisup
                
                % init semisup chains at mean training set values
                % not sure if that is correct?
                meany  = samplematrix(repmat(mean(opts.y_train,1),opts.batchsize,1));
                chains = [opts.x_semisup_batch; v0;];
                chainsy = [meany;ey;];
            else
                chains = v0;
                chainsy = ey;
            end
            init_chains = 0;
        end
        
        if rbm.dropout_hidden > 0
            rbm.hidden_mask = (rbm.rand(size(n_hidden,opts.batchsize)) > rbm.dropout_hidden);
        end
        
        % calculate rbm gradients
        [grads,c_err,chains,chainsy]= rbm.train_func(rbm,v0,ey,opts,chains,chainsy);
        
        err = err + c_err;
        %fprintf('%d\n',c_err)
        
        %update weights, LR,decay and momentum
        rbm = rbmapplygrads(rbm,grads,v0,ey,epoch);
    end
    rbm.error(end+1) = err / numbatches;
    
    % calc train/val performance.
    [perf,rbm] = rbmmonitor(rbm,x_train,opts,x_samples,val_samples,epoch);
    earlystop  = rbmearlystopping(rbm,opts,earlystop,epoch);
    
    % stop training?
    if rbm.early_stopping && earlystop.patience < 0
        disp('No more Patience. Return best RBM')
        earlystop.best_rbm.val_error = rbm.val_error;
        earlystop.best_rbm.train_error = rbm.train_error;
        earlystop.best_rbm.error = rbm.error;
        rbm = earlystop.best_rbm;
        
        break;
    end
% display output
epochnr = ['Epoch ' num2str(epoch) '/' num2str(opts.numepochs) '.'];
avg_err = [' Avg recon. err: ' num2str(err / numbatches) '|'];
lr_mom  = [' LR: ' num2str(rbm.curLR) '. Mom.: ' num2str(rbm.curMomentum)];
disp([epochnr avg_err lr_mom perf earlystop.best_str]);    
end
%rbm.c = rbm.c - rbm.W*rbm.xt_MU';   %ZM algorithm



end




