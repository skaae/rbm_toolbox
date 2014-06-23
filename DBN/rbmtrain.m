function rbm = rbmtrain(rbm, x, opts)
%%RBMTRAIN trains a single RBM
% notation:
%    w  : weights
%    b  : bias of visible layer
%    c  : bias of hidden layer
assert(isfloat(x), 'x must be a float');
assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
m = size(x, 1);
numbatches = m / opts.batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches not integer');


init_chains = 1;
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
        
        switch opts.traintype
            
            case 'CD'
                %contrastive divergence sampling
                [dW,db,dc,curr_err] = cdk(rbm,v0,opts.cdn);
            case 'PCD'
                % persistent CD, note that we reuse the current state of 
                % of the chains
                [dW,db,dc,curr_err,chains] = pcd(rbm,v0,opts.cdn,chains);
            otherwise
                error('opts.traintype must be CD|PCD')
        end
        
        
        
        
        rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * dW / opts.batchsize;
        rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * db / opts.batchsize;
        rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * dc / opts.batchsize;
        
        rbm.W = rbm.W + rbm.vW;
        rbm.b = rbm.b + rbm.vb;
        rbm.c = rbm.c + rbm.vc;
        
        err = err + curr_err / opts.batchsize;
    end
    
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
    
end
end
