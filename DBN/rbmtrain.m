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
m = size(x_train, 1);
numbatches = m / opts.batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches not integer');

% use validation set or not
if ~isempty(opts.x_val)
    use_val = 1;
    samples = randperm(size(x_train,1));
    val_samples = samples(1:size(opts.x_val,1));  
end

% RUN epochs
init_chains = 1;
chains = [];
chainsy = [];

for i = 1 : opts.numepochs
    kk = randperm(m);
    err = 0;
    for l = 1 : numbatches
        v0 = x_train(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
        
        if rbm.hintonDBN == 1
            ey = opts.y_train(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
        else
            ey = [];
        end
        
        
        if strcmp(opts.traintype,'PCD') && init_chains == 1
            % init chains in first epoch if Persistent contrastive divergence
            chains = v0;
            chainsy = ey;
            init_chains = 0;
        end
        
        % Collect rbm statistics with CD or PCD
        [dw,db,dc,du,dd,c_err,chains,chainsy] = rbmstatistics(rbm,v0,ey,opts,...
                                                            chains,chainsy);

        
        err = err + c_err;
        
        %update weights, LR,decay and momentum 
        rbm = rbmapplygrads(rbm,dw,db,dc,du,dd,v0,ey,i);
    end
    rbm.error(end+1) = err / numbatches;

    % if the training data energy is much lower than the validation energy
    % rasie a overfitting warning. (i.e the ratio becomes <1)
    if mod(i,opts.ratio_interval) == 0 && use_val == 1
        rbm = rbmoverfitting( rbm,x_train,val_samples,opts,i);
        overfit = ifelse(rbm.ratioy(end)<0.8,'(overfitting)','(OK)');
        energy = sprintf('. E_Val / E_train %4.3f %s',rbm.ratioy(end),overfit);
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
