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

    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        for l = 1 : numbatches
            v1 = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            
            switch opts.traintype
            
                case 'CD' 
                    %contrastive divergence sampling
                    [v2,h1,h2] = cdn(rbm,v1,opts.cdn);
                case 'PCD'
                    %persistent CD se
                    error('Not implemented')
                otherwise
                    error('opts.traintype must be CD|PCD')
            end


            c1 = h1' * v1;
            c2 = h2' * v2;

            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
end
