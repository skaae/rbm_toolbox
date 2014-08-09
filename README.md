**THIS DOCUMENTATION DOES NOT CORRESPOND WITH CURRENT CODE**

# RBM Toolbox

RBM toolbox is a MATLAB toolbox for online training of RBM and stacked RBM's.

 * Support for training RBM's with class labels including:
    * Generative training objective [2,7]
    * Discriminative training objective [2,7]
    * Hybrid training objective [2,7]
    * Semi-supervised learning [2,7]
 * CD - k (contrastive divergence k) [5]
 * PCD (persistent contrastive divergence) [6]
 * RBM/DBN sampling functions (pictures / movies)
 * RBM/DBN Classification support [2,7]
 * Regularization: L1, L2, sparsity, early-stopping, dropout [1],dropconnect[10], momentum [3] 

The code in the toolbox is partly based on the DeepLearnToolbox by Rasmus Berg Palm. 


This README first describes settings in the toolbox. Usage examples are given afterwards. Note that a single layer DBN is a RBM. 

# Settings
Settings in the toolbox are generally controlled with the `opts` struct. An `opts` struct with default values can be created with

```MATLAB
opts = dbncreateopts();
```

The DBN network is then created and trained with:

```MATLAB
sizes = [50]
dbn = dbnsetup(sizes,x_train,opts);
dbn = dbntrain(dbn,x_train,opts);
```

Here sizes specifies the sizes of the hidden layers in the RBM's. In the example a single RBM with 50 hidden units is created. If sizes is is a vector, e.g `sizes = [50 100]`, a RBM with 50 hidden units and a RBM with 100 hidden units are stacked. Sizes of visible layers are inferred from the data.

## Training Objectives
The toolbox support three different training objectives:

 * Generative    : optimizes `-log( p(x,y) )` 
 * Discrminative : optimizes `-log( p(y | x) )`
 * Hybrid        : optimizes `-alpha * log( p(x,y) ) - (1-alpha) * log( p(y | x) )`

Furthermore semisupervised training is available. In semisupervised training unlabled data is used in conjunction with the labeled data. 


RBM weights are updated using the equation:

`grads = alpha * grads_generative + (1-alpha) * grads_discriminative  + beta * grads_semisupervised`

From the equation it follows that:

 * `opts.alpha = 0` is discriminative training
 * `opts.alpha = 1` is generative training
 * `opts.alpha = ]0;1[` is hybrid training

 Semisupervised training is added by setting `opts.beta > 0`. In semisupervised training we partly train on unlabeled data. 
 The toolbox uses y values sampled from `p(y|x)` as labels for the semisupervised samples.  The procedure is described in [7].

## Specifying training, validation and semisupervised data

 * `x_train`        : pass x training data to the toolbox with `dbntrain(dbn,x_train,opts)`
 * `opts.y_train`   : y training data  
 * `opts.x_val`     : x validation data (optional)
 * `opts.y_val`     : y validation data (optional)
 * `opts.x_semisup` : x semisupervised data (optional)

 During training training and validatiion performance is monitored. The number of epochs between calcualtion of training and validaiton performance is controlled with `opts.testinterval`.

 If the traning set is large it might be costly to evaluate the training performance using the whole training set. `opts.traintestbatch` controls the number of samples used for calculation of training performance. 

## Learning rate and momentum

The learning rate is controlled with `opts.learningrate`. `opts.learningrate` should be a handle to a function taking current epoch and momentum as input, this allows for decaying learning rate.

Learning rate is set with:   

```MATLAB
% Decaying learning rate, t = current epoch
eps               = 0.001;    % initial learning rate
f                 = 0.99;      % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);

% Constant learning rate
opts.learningrate =  0.01;
```

Momentum is controlled through `opts.momentum`. `opts.momentum` should be a function taking current epoch as input.

Momentum is set with:

```MATLAB
% Momentum with ramp-up, t = current epoch
T             = 50;     % momentum ramp up epoch
p_f           = 0.9;    % final momentum
p_i           = 0.5;    % initial momentum
opts.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f, p_f);

% Constant momentum
opts.momentum =  0.9;
```

## Sampling statistics

The toolbox support Contrastive divergence (`CD`)[5] and persistent contrastive divergcence (`PCD`) [6] for collecting statistics. 

 * Contrastive divergence: `opts.traintype = 'CD'`
 * Persistent contrastive divergence:  `opts.traintype = 'PCD'`

The number of Gibbs steps before the statistics is collected is controlled with `opts.cdn`. `opts.cdn` is eiether a scalar in wich case the same number of gibbs steps is used for all epochs. 

Choose the sampling method with `opts.traintype`. For `PCD` the number of persistent chains is controlled with `opts.npcdchains`. 
 `opts.npcdchains` must be less than the the number of samples and the number of semisupervised samples, the default number of chains is 100.

 The number of Gibbs steps before the negative statistics is collected is controlled with `opts.cdn`. `opts.cdn` is specified similarly to the learning rate and momentum, i.e:

```MATLAB
% Increasing gibbs steps at fifth epoch
T = 5;
opts.cdn = @(epoch) ifelse(t < T,1,5);

% Constant CDn
opts.cdn =  1;
```

## Initial Weight Sizes

Initial weights are either sampled from a normal distribution [3] or from a uniform distribution [7], the behavior is controlled through `opts.init_type`:

 * `opts.init_type = 'gauss'`    : N(0, 0.01)
 * `opts.init_type = 'cRBM'`     : Unif(-M^-0.5, M^-0.5), is max(n_columns, n_rows) of weight matrix. 
 * `opts.init_type = @(m,n) func : Handle to funtion returning a M-by-N matrix.

## Regularization

The following regularization options are implemented:

 * `opts.L1`: specify the regularization weight
 * `opts.L2`: specify the regularization weight
 * `opts.sparsity`: implemented as in [7]. Specify the sparsity being subtracted from biases after each weight update.
 * `opts.dropout`: dropout on hidden units. Specify the 1-probability of being dropped. see [1]
 * `opts.dropconnect`: dropout on connections, specify 1-probability of connection being zeroed, see [10]
 * Early-stopping

#### Dropout Weights
In dropout the hidden units are dropped with `1-opts.dropout`. During each weight update rows of the incoming weights and biases to the hidden units are clamped to zero. W is the weights between visible and hiiden units. In dropout rows of W are clamped to zero. The picture below shows the dropout W (left) and the original W (right). Black is a wieght value equal to zero and white is a weight value > 0.
<html>
<img src="/uploads/dropout.png" height="200" width="500"> 



#### DropConnect Weights
DropConnect drops connections between visible and hidden units with probability `1-opts.dropconnect`. The picture shows W with dropconnect enabled (left) and the original weights (right)
<html>
<img src="/uploads/dropconnect.png" height="200" width="500"> 

####
Early stopping is always enabled. The early stopping patience is set with `opts.patience`. If you want to disable early stopping set `opts.patience = Inf`. 

## Using the CPU / GPU

`opts.gpu` switches between CPU and GPU. GPU requires the MATLAB Parallel Computing Toolbox.

 * `opts.gpu = 1` : Use GPU. Requires that `opts.thisgpu` is set to a reference to the selected GPU (use `opts.thisgpu = gpuDevice()`).
 * `opts.gpu = 0` : Use CPU. 
 * `opts.gpu = -1`: For testing

# Examples

Reproducing results from [7], specifically the results from the table reproduced below:

| Model  |Objective                                         | Errror (%)    | Example  |
|---     |---                                               |---            |---       |
|        | Generative(lr = 0.005, H = 6000)                 |   3.39        |    4     |
|ClassRBM| Discriminative(lr = 0.05, H = 500)               |   1.81        |    5     |
|        | Hybrid(alpha = 0.01, lr = 0.05, H = 1500)        |   1.28        |    6     |
|        | Sparse Hybrid( idem + H = 3000, sparsity=10^-4)  |   1.16        |    7     |
lr = learning rate
H = hidden layer size



The networks were trained on the MNIST data set.
To reproduce the results the following settings where used:
 * online learning, i.e a batchsize of 1. 
 * Initial weights are taken from uniform samples in the interval [-m^(-0.5), m^(-0.5)] where m is max([n_rows, n_columns)] of the matrix being initilaized. 
 * Early stopping with patience of 15.
 * MNIST training set was randomly split into a training set of 50000 samples and a validation 10000 samples. The original test set was used. 

Weight initalization is important, try experiment by supplying your own initalization functions. 


## Example 4 - Discriminative 

```MATLAB
%% Example 4 - Discriminative training
% Tries to reproduce discriminative result from table 1 in 
% "Learning algorithms for the classification Restricted boltzmann machine"
name = 'example_4';
rng('default');rng(101);
 [train_x,val_x,test_x,train_y,val_y,test_y] = setupmnist(101,1);
f = fullfile(pwd,[name '.mat'])

% Setup DBN
sizes = [500 ];   % hidden layer size

opts = dbncreateopts();
opts.alpha = 0; % 0 = discriminative, 1 = generative
opts.beta = 0;
opts.gpu   = 0;                  % use GPU other optsion are 0: CPU, -1: CPU test
opts.cdn = 1;   
opts.thisgpu = [];              % ref to gpu,  must be set if opts.gpu =1
opts.gpubatch = size(train_x,1); 
opts.outfile = [name '_intermediate.mat'];
opts.patience = 15;
opts.numepochs = 1000;
opts.testinterval = 1;
opts.init_type = 'cRBM';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = val_x;
opts.y_val = val_y;


%% Set learningrate and momentum
opts.learningrate = @(t,momentum) 0.05;
opts.momentum = @(t) 0;


[dbn, opts]  = dbnsetup(sizes, train_x, opts);  % train function 

rbm = dbn.rbm{1};
opts.gpu
opts.numepochs
disp(rbm);

fprintf('\n\n')

rbm = rbmtraingpu(rbm,train_x,opts);

save(f,'rbm','opts');
```

## Example 5 - Generative

```MATLAB
%% Example 5 - generative training
% Tries to reproduce discriminative result from table 1 in 
% "Learning algorithms for the classification Restricted boltzmann machine"
name = 'example_5';
rng('default');rng(101);
 [train_x,val_x,test_x,train_y,val_y,test_y] = setupmnist(101,1);
f = fullfile(pwd,[name '.mat'])

% Setup DBN
sizes = [6000 ];   % hidden layer size
opts = dbncreateopts();
opts.alpha = 1; % 0 = discriminative, 1 = generative
opts.beta = 0;
opts.gpu   = 0;                  % use GPU other optsion are 0: CPU, -1: CPU test
opts.cdn = 1;   
opts.thisgpu = [];              % ref to gpu,  must be set if opts.gpu =1
opts.gpubatch = size(train_x,1); 
opts.outfile = [name '_intermediate.mat'];
opts.patience = 4;
opts.numepochs = 1000;
opts.testinterval = 5;
opts.init_type = 'cRBM';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = val_x;
opts.y_val = val_y;


%% Set learningrate and momentum
opts.learningrate = @(t,momentum) 0.005;
opts.momentum = @(t) 0;


[dbn, opts]  = dbnsetup(sizes, train_x, opts);  % train function 

rbm = dbn.rbm{1};
opts.gpu
opts.numepochs
disp(rbm);

fprintf('\n\n')

rbm = rbmtraingpu(rbm,train_x,opts);
save(f,'rbm','opts');
```

## Example 6 - Hybrid training 

```MATLAB
%% Example 6 - Hybrid
% Tries to reproduce discriminative result from table 1 in 
% "Learning algorithms for the classification Restricted boltzmann machine"
name = 'example_6';
rng('default');rng(101);
 [train_x,val_x,test_x,train_y,val_y,test_y] = setupmnist(101,1);
f = fullfile(pwd,[name '.mat'])

% Setup DBN
sizes = [1500 ];   % hidden layer size
opts = dbncreateopts();
opts.alpha = 0.01; % 0 = discriminative, 1 = generative
opts.beta = 0;
opts.gpu   = 0;                  % use GPU other optsion are 0: CPU, -1: CPU test
opts.cdn = 1;   
opts.thisgpu = [];              % ref to gpu,  must be set if opts.gpu =1
opts.gpubatch = size(train_x,1); 
opts.outfile = [name '_intermediate.mat'];
opts.patience = 15;
opts.numepochs = 1000;
opts.testinterval = 1;
opts.init_type = 'cRBM';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = val_x;
opts.y_val = val_y;


%% Set learningrate and momentum
opts.learningrate = @(t,momentum) 0.05;
opts.momentum = @(t) 0;


[dbn, opts]  = dbnsetup(sizes, train_x, opts);  % train function 

rbm = dbn.rbm{1};
opts.gpu
opts.numepochs
disp(rbm);

fprintf('\n\n')

rbm = rbmtraingpu(rbm,train_x,opts);
save(f,'rbm','opts');
```


## Example 7 - Hybrid training with sparsity

```MATLAB
%% Example 7 - Sparse hybrid
% Tries to reproduce discriminative result from table 1 in 
% "Learning algorithms for the classification Restricted boltzmann machine"
name = 'example_6';
rng('default');rng(101);
 [train_x,val_x,test_x,train_y,val_y,test_y] = setupmnist(101,1);
f = fullfile(pwd,[name '.mat'])

% Setup DBN
sizes = [3000 ];   % hidden layer size
opts = dbncreateopts();
opts.alpha = 0.01; % 0 = discriminative, 1 = generative
opts.beta = 0;
opts.gpu   = 0;                  % use GPU other optsion are 0: CPU, -1: CPU test
opts.cdn = 1;   
opts.sparsity = 10^-4;
opts.thisgpu = [];              % ref to gpu,  must be set if opts.gpu =1
opts.gpubatch = size(train_x,1); 
opts.outfile = [name '_intermediate.mat'];
opts.patience = 15;
opts.numepochs = 1000;
opts.testinterval = 1;
opts.init_type = 'cRBM';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = val_x;
opts.y_val = val_y;


%% Set learningrate and momentum
opts.learningrate = @(t,momentum) 0.05;
opts.momentum = @(t) 0;


[dbn, opts]  = dbnsetup(sizes, train_x, opts);  % train function 

rbm = dbn.rbm{1};
opts.gpu
opts.numepochs
disp(rbm);

fprintf('\n\n')

rbm = rbmtraingpu(rbm,train_x,opts);
save(f,'rbm','opts');
```


# TODO

 * Add Annealed Importance Sampling (AIS) [8]
 * add Normalization to RBM [9]
 *  Parallel tempering for training of restricted Boltzmann machines.
# 

# References

[1] N. Srivastava and G. Hinton, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” J. Mach.  …, 2014.  
[2] H. Larochelle and Y. Bengio, “Classification using discriminative restricted Boltzmann machines,” … 25th Int. Conf. Mach. …, 2008.  
[3] G. Hinton, “A practical guide to training restricted Boltzmann machines,” Momentum, 2010.  
[4] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov, “Improving neural networks by preventing co-adaptation of feature detectors,” Jul. 2012.  
[5] G. Hinton, “Training products of experts by minimizing contrastive divergence,” Neural Comput., 2002.  
[6] T. Tieleman, “Training restricted Boltzmann machines using approximations to the likelihood gradient,” … 25th Int. Conf. Mach. …, 2008.  
[7] H. Larochelle and M. Mandel, “Learning algorithms for the classification restricted boltzmann machine,” J. Mach.  …, 2012.
[8] R. Salakhutdinov and I. Murray, “On the quantitative analysis of deep belief networks,” …  25th Int. Conf. …, 2008.   
[9] Y. Tang and I. Sutskever, “Data normalization in the learning of restricted Boltzmann machines,” 2011.  
[10] L. Wan, M. Zeiler, S. Zhang, Y. Le Cun, and R. Fergus, “Regularization of Neural Networks using DropConnect,” in Proceedings of The 30th International Conference on Machine Learning, 2013, pp. 1058–1066. 

Copyright (c) 2014, Søren Kaae Sønderby (skaaesonderby@gmail.com) All rights reserved.