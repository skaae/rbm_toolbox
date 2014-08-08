function batches = createminibatches(batchsize,x)
%CREATEMINITACHES creates a list of minibatches start and ends
%   INPUTS
%        batchsize    : minibatch size
%                x    : data
%
% Copyright Søren Sønderby August 2014



n_samples = size(x,1);

if n_samples > 0
    if batchsize > n_samples
        batchsize = n_samples;
    end
    
    numbatches = ceil(n_samples / batchsize);
    batches = zeros(numbatches,2);
    for batch_num = 1:numbatches
        batch_start = (batch_num - 1) * batchsize + 1;
        batch_end = batch_num * batchsize;
        
        if (batch_end + batchsize) <= n_samples
            batch = [batch_start, batch_end];
        else
            batch = [batch_start, n_samples];
        end
        batches(batch_num,:) = batch;
    end
else
    batches = [];
end




end