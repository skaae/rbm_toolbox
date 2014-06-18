function test_example_NN
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rand('state',0)
nn = nnsetup([784 100 10]);
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.08, 'Too big error');

%% ex2 neural net with L2 weight decay
rand('state',0)
nn = nnsetup([784 100 10]);

nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');


%% ex3 neural net with dropout
rand('state',0)
nn = nnsetup([784 100 10]);

nn.dropoutFraction = 0.5;   %  Dropout fraction 
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex4 neural net with sigmoid activation function
rand('state',0)
nn = nnsetup([784 100 10]);

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.learningRate = @(epoch) 1;                %  Sigm require a lower learning rate
opts.numepochs =  1;                %  Number of full sweeps through data
opts.batchsize = 100;               %  Take a mean gradient step over this many samples

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex5 plotting functionality
rand('state',0)
nn = nnsetup([784 20 10]);
opts.numepochs         = 5;            %  Number of full sweeps through data
nn.output              = 'softmax';    %  use softmax output
opts.batchsize         = 1000;         %  Take a mean gradient step over this many samples
opts.plot              = 1;            %  enable plotting

nn = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');

%% ex6 neural net with sigmoid activation and plotting of validation and training error
% split training data into training and validation data
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);

rand('state',0)
nn                      = nnsetup([784 20 10]);     
nn.output               = 'softmax';                   %  use softmax output
opts.numepochs          = 5;                           %  Number of full sweeps through data
opts.batchsize          = 1000;                        %  Take a mean gradient step over this many samples
opts.plot               = 1;                           %  enable plotting
nn = nntrain(nn, tx, ty, opts, vx, vy);                %  nntrain takes validation set as last two arguments (optionally)

[er, bad] = nntest(nn, test_x, test_y);
assert(er < 0.1, 'Too big error');


% ex7 vanilla net with exponentially decaying learning rate
rand('state',0)
nn = nnsetup([784 100 10]);
nn.learningRate = @(epoch) 2*exp(-0.5*epoch); 
opts.numepochs =  5;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.08, 'Too big error');

% ex8 vanilla net with increasing momentum
rand('state',0)
nn = nnsetup([784 100 10]);
opts.plot=0;
nn.momentum = @(epoch) ifelse(epoch < 5,0.1*epoch,0.5); 
opts.numepochs =  10;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.08, 'Too big error');




% Hinton replication

rand('state',0)
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);



nn = nnsetup([784 1200 1200 10]);
opts.plot=1;
T = 500;    % momentum ramp up
p_f = 0.99;  % final momentum
p_i = 0.5;    % initial momentum
eps = 10;   % initial learning rate
f = 0.998;  %learning rate decay
nn.learningRate = @(t,momentum) eps.*f.^t*(1-momentum); 
nn.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f); 
opts.numepochs =  3000;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
nn.dropoutFraction                  = 0.5; 
nn.weightPenaltyL2                  = 0; 
nn.weightMaxL2norm                  = 15;

%init weights with zero mean and sigma 0.01
    for i = 2 : nn.n   
        % weights and weight momentum
        nn.W{i - 1} = normrnd(0,0.01,size(nn.W{i - 1}));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end


[nn, L] = nntrain(nn, train_x, train_y,opts,test_x,test_y);

[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.08, 'Too big error');






