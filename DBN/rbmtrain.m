function rbm = rbmtrain(rbm, x, opts)
%%RBMTRAIN trains a single RBM
% notation:
%    w  : weights
%    b  : bias of visible layer
%    c  : bias of hidden layer
%  Modified by Søren Sønderby June 2014
assert(isfloat(x), 'x must be a float');
assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
m = size(x, 1);
numbatches = m / opts.batchsize;
assert(rem(numbatches, 1) == 0, 'numbatches not integer');


init_chains = 1;
chains = [];
for i = 1 : opts.numepochs
    kk = randperm(m);
    err = 0;
    for l = 1 : numbatches
        v0 = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
        
        if strcmp(opts.traintype,'PCD') && init_chains == 1
            % init chains in first epoch if Persistent contrastive divergence
            chains = v0;
            init_chains = 0;
        end
        
        % Collect rbm statistics with either CD or PCD
        [dw,db,dc,c_err,chains] = rbmstatistics(rbm,v0,opts,opts.traintype,chains);
        
        %update weights, LR and momentum
        rbm = rbmapplygrads(rbm,dw,db,dc,i)
        

        
        err = err + c_err / opts.batchsize;
    end
    
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
    
end
end
