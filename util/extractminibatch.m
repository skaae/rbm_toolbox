function batch = extractminibatch(kk,minibatch_num,batchsize,x)
%EXTRACTMINIBATCH extract minibatch
%   INPUTS
%               kk    : random permuation of i.e   kk = randperm(n_samples);
%     minibatchnum    : current minibatch
%        batchsize    : minibatch size
%                x    : data 
batch_start = (minibatch_num - 1) * batchsize + 1;
batch_end = minibatch_num * batchsize;
n_samples = size(x,1);
if (batch_end + batchsize) <= n_samples
        %batch_x = train_x( kk((l - 1) * batchsize + 1 : l * batchsize) , :);
    idx = kk(batch_start:batch_end);
        batch = x( idx,:);
else
    batch = x( kk(batch_start:end),:);
end
end