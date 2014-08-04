function [train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist(seed,downsize)
% loads mnist. Creates training, validation and test set
% specify random seet with seed, if no seed is given 0 is used
% specify downsizing of datasets. I.e 0.5 means 50%, 0.1 means 10% of original
% dataset.
load mnist_uint8;

if ~exist('seed','var')
    seed = 0;
end

rng('default');rng(seed);
n_samples = size(train_x,1);
s = randperm(n_samples);
val_samples  = s(1:10000);
train_samples = s(10001:end);


% Test set
test_x  = double(test_x)/255;
test_y = double(test_y);

%Training and validation set
train_x = double(train_x)/255;
train_y = double(train_y);

val_x   = train_x(val_samples,:);
train_x = train_x(train_samples,:);
val_y   = train_y(val_samples,:);
train_y = train_y(train_samples,:);


% resize if downsize is given
if exist('downsize','var')
    if downsize > 1 || downsize < 0
        error('downsize must lie in ]0,1]');
    else
        resize = @(dataset) dataset(1:floor(size(dataset,1)*downsize),:);
        train_x = resize(train_x);
        val_x = resize(val_x);
        test_x = resize(test_x);
        train_y = resize(train_y);
        val_y = resize(val_y);
        test_y = resize(test_y);
    end
    
end