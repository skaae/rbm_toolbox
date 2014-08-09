**THIS DOCUMENTATION DOES NOT CORRESPOND WITH CURRENT CODE**

# RBM Toolbox

RBM toolbox is a MATLAB toolbox for training RBM's.

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


This README first describes settings in the toolbox. Usage examples are given afterwards.

## Training Objectives
The toolbox support three different training objectives:

 * Generative training: optimizes -log(p(x,y))`
 * Discrminative training: optimizes  `p(y I x)`

 &#9824

Training objectives are controlled with:

 * opts.alpha: Controls the weight for generative training. 1 is pure generative, 0 is pure discriminative, intermediate values are hybrid training
 * opts.beta : Controls semisupervied weight. 0 is no semisupervised, you must supply opts.x_semisup for opts.beta > 0 


#### Generative training
Generative training and discriminative training are the basic training objectives in the RBM toolbox. The hybrid and semi-supervised training objectives are different combinations of generative and discriminative training. In generative training the objective `-log(p(x,y))` is optimized.

#### Discriminative training
Discriminative training optimizes `p(y I x)`.


#### Semi-supervised training
Semi-supervised training combines unsupervised and supervised training. The following formulae is used to combine the objectives:

`grads = grads + opts.semisup_beta * grads_generative`

Here `grads` is the gradients obtained from the labeled samples. We use samples of p(y I x) as labels for the unsupervised training.

## Learning rate and momentum

The learning rate is controlled with `opts.learningrate`. `opts.learningrate` should be a handle to a function taking current epoch and momentum as input, this allows for decaying learning rate.

Decaying learning rate can be specified with:   

```MATLAB
eps               = 0.001;    % initial learning rate
f                 = 0.99;      % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);
```
Constant learning rate can be spcified with

```MATLAB
opts.learningrate = @(t,momentum) 0.01;
```

Momentum is controlled through `opts.momentum`. `opts.momentum` should be a function taking current epoch as input.

Ramp up momentum can be specified with:

```MATLAB
T             = 50;       % momentum ramp up epoch
p_f           = 0.9;    % final momentum
p_i           = 0.5;    % initial momentum
opts.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);
```

Constant momentum can be specified with:

```MATLAB
opts.momentum = @(t) 0.9;
```

## Initial Weight Sizes

Initial weights are either sampled from a normal distribution [3] or from a uniform distribution [7], the behavior is controlled through `opts.init_type`:

```MATLAB
switch lower(opts.init_type)
    case 'gauss'
        initfunc = @(m,n) normrnd(0,0.01,m,n);
    case 'crbm'
        initfunc = @(m,n) init_crbm;
    otherwise
        error('init_type should be either gauss or cRBM');
end

    function weights = init_crbm(m,n)
        M = max([m,n]);
        interval_max = M^(-0.5);
        interval_min = -interval_max;
        weights = interval_min + (interval_max-interval_min).*rand(m,n);
    end
```
Lastly you can supply your own function as a handle to `opts.init_type`. The signature must be `f(n_rows,n_columns)`. The output should be a dobule m-by-n matrix. e.g:

```MATLAB
opts.init_type = @(m,n)  normrnd(0,100,m,n);
```



## Regularization

The following regularization options are implemented

 * `opts.L1`: specify the regularization weight
 * `opts.L2`: specify the regularization weight
 * `opts.sparsity`: implemented as in [7]. Specify the sparsity being subtracted from biases after each weight update.
 * `opts.dropout`: dropout on hidden units. Specify the probability of being dropped. see [1]
 * `opts.dropconnect`: dropout on connections see [10]


**Dropout Weights**  
<html>
<img src="/uploads/dropout.png" height="200" width="500"> 

**DropConnect Weights**  
<html>
<img src="/uploads/dropconnect.png" height="200" width="500"> 


** Early Stopping**  
Early stopping is always enabled. The early stopping patience is set with opts.patience. If you want to disable early stopping set `opts.patience = Inf`. 

## Hidden Layer Sizes

In the example a RBM with 500 hidden units is created and trained. You do not need to set the size of the visible units.

```MATLAB
sizes = [500];                        % hidden layer size
opts = dbncreateopts();   % create default opts struct
dbn = dbnsetup(sizes, train_x, opts);     % create dbn struct
rbm = dbn.rbm{1};
rbm = rbmtrain(rbm, train_x, opts);       % train  rbm
```
## Using the GPU

To use GPU set `opts.gpu = 1`. Setting `opts.gpu = 0` uses CPU and `opts.gpu = -1` is for testing. 
When `opts.gpu`is 1 then `opts.thisgpu` must be set to `gpuDevice()`. 


## Sampling statistics
The toolbox support Contrastive divergence (`CD`)[5] and persistent contrastive divergcence (`PCD`) [6] for collecting statistics. 
Choose the sampling method with `opts.traintype`. For `PCD` the number of persistent chains is controlled with `opts.npcdchains`. 
 `opts.npcdchains` must be less than the the number of samples and the number of semisupervised samples, the default number of chains is 100. 

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