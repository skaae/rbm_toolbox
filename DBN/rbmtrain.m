function rbm = rbmtrain(rbm, x_train,opts)
%%RBMTRAIN trains a single RBM
% notation:
% data  : all data given as      [n_samples   x #vis]
%    W  : vis - hid weights      [ #hid       x #vis ]
%    U  : label - hid weights    [ #hid       x #n_classes ]
%    b  : bias of visible layer  [ #vis       x 1]
%    c  : bias of hidden layer   [ #hid       x 1]
%    d  : bias of label layer    [ #n_classes x 1]
%  Modified by Søren Sønderby June 2014

% SETUP and checking
assert(isfloat(x_train), 'x must be a float');
assert(all(x_train(:)>=0) && all(x_train(:)<=1), 'all data in x must be in [0:1]');
n_samples = size(x_train, 1);
[n_hidden, n_visible] = size(rbm.W);

numbatches = n_samples / opts.batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches not integer');

% use validation set or not
if ~isempty(opts.x_val)
    n_val_samples   = size(opts.x_val,1);
    samples         = randperm(size(x_train,1));
    % if size of val set is larger than train set use trainset size otherwise
    % use size of validation set
    size_val_sample = ifelse(n_samples>=n_val_samples, n_val_samples, n_samples);
    val_samples     = samples(1:size_val_sample);
else
    val_samples = [];
end

if rbm.early_stopping
    best_error = Inf;
    patience  = rbm.patience;
end

% RUN epochs
init_chains = 1;
chains = [];
chainsy = [];
best = '';

for epoch = 1 : opts.numepochs
    kk = randperm(n_samples);
    err = 0;
    for l = 1 : numbatches
        v0 = extractminibatch(kk,l,opts.batchsize,x_train);
        if rbm.classRBM == 1
            ey = extractminibatch(kk,l,opts.batchsize,opts.y_train);
        else
            ey = [];
        end
        
        
        if strcmp(opts.traintype,'PCD') && init_chains == 1
            % init chains in first epoch if Persistent contrastive divergence
            chains = v0;
            chainsy = ey;
            init_chains = 0;
        end
        % calculate rbm gradients
        [grads,c_err,chains,chainsy]= opts.train_func(rbm,v0,ey,opts,chains,chainsy);
        
        err = err + c_err;
        
        %update weights, LR,decay and momentum
        rbm = rbmapplygrads(rbm,grads,v0,ey,epoch);
    end
    rbm.error(end+1) = err / numbatches;
    
    % calc train/val performance.
    [perf,rbm] = rbmmonitor(rbm,x_train,opts,val_samples,epoch);
    
    if rbm.early_stopping && ~isempty(rbm.val_error)
        if best_error > rbm.val_error(end)
            best = ' ***';
            best_error = rbm.val_error(end);
            best_rbm = rbm;
            patience = rbm.patience;
        else
            best = '';
            patience = patience-1;
        end
        % stop training
        if patience < 0
            disp('No more Patience. Return best RBM')
            rbm = best_rbm;
            break;
        end
    end
    
    % display output
    epochnr = ['Epoch ' num2str(epoch) '/' num2str(opts.numepochs) '.'];
    avg_err = [' Avg recon. err: ' num2str(err / numbatches) '|'];
    lr_mom  = [' LR: ' num2str(rbm.curLR) '. Mom.: ' num2str(rbm.curMomentum)];
    disp([epochnr avg_err lr_mom perf best]);
    
    
end



end
