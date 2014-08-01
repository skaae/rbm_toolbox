function [train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist()

if ~ismac
    cd('../..');
    addpath(genpath(pwd()));
end

%% setup training
rng('default');rng(0);
load mnist_uint8;

% Test set
test_x  = double(test_x)/255;
test_y = double(test_y);

%Training and validation set
train_x = double(train_x)/255;
train_y = double(train_y);

val_x   = train_x(1:10000,:);
train_x = train_x(10001:end,:);
val_y   = train_y(1:10000,:);
train_y = train_y(10001:end,:);