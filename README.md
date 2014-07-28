# RBM Toolbox

RBM toolbox is a MATLAB toolbox for training RBM's. It builds on the DeepLearnToolbox by Rasmus Berg.

RBM toolbox support among others:

 * add support for training RBM's with class labels including, see [1,7]
    * generative training objective
    * discriminative training objective
    * hybrid training objective
    * semi-supervised learning
 * CD - k (contrastive divergence k)
 * PCD (persistent contrastive divergence)
 * Various rbm sampling functions (pictures / movies)
 * Classiciation support
 * Regularization: L1, L2, maxL2norm, sparsity, early-stopping
 * Support for custom error functions


# Settings
The following section describes som of the options. Refer to `dbncreateopts` for a description of all settings.

## training objectives

The RBM toolbox supports four different TBM training objectives. For a detailed description refer to [7].

The RBM training objective is set by supplying a function handle to one of the four training functions through `opts.train_func`.
When `opts.classRBM==1` the toolbox will use both x and y as input to the visible units. You specify y training samples with `opts.y_train` and validation validation samples with `opts.x_val` and `opts.y_val`. 

####Training objectives

* `rbmgenerative`:  `-log(p(x))` or `-log(p(x,y))` if `classRBM` is 1
* `rbmdiscriminative`  `-log(p(x I y))`   [7]
* `rbmhybrid` Models   `-(1-alpha)log(y I x) - alpha log(p (x) )`. The importance of the generative traning objective is 
controleld with `opts.hybrid_alpha` [7]
* `rbmsemisuplearn`    `TYPE + beta unsupervised. Where type is {generative, discriminative,hybrid} and unsupervised is generative training on unlabeled data. The importance of unsupervised training is controlled with `opts.semisup_beta`. TYPE is specified with `opts.semisup_type = traintype` where `traintype is `{@rbmgenerative, @rbmdiscriminative, @rbmhybrid}`. Samples used for unsupervised training are given with `opts.x_semisup`.  See [7].

#### Generative training
Generative training and discrmininaive training are the basic training objectives in the toolbox. The hybrid and semisupervised training objectives are different combinations of generative and discriminative training. In generative training the negative log likelihood is optimized, depending on the seeting of opts.classRBM that is either `-log(p(x))` or `-log(p(x,y))`.

Generative training can be trained with contrastive divergence (`CD`) [5] or with persistent contrastive divergence (`PCD`) [6]. The traning type is specified with `opts.traintype`. Both CD and PCD samples negative statistics using gibbs sampling. The number of gibbs steps before the statistics is collected is specified with `opts.cdn`. Usually 1 works fine. 

#### Discriminative training
Discriminative training optimizes `p(y I x)`. 

#### Hybrid training
Hybrid training combines generative and discriminative training objective. The two objectives are combined with:

`grads = grads_discrminative + opts.hybrid_alpha X grads_generative` 

### Semisupervised training
Semisupervised training combines unsupervised and supervised training. The following formulae is used to combine the objectives:

`grads = grads_type + opts.semisup_beta * grads_generative`

Here `grads_type` is either the gradients from generative, discriminative or hybrid training. `grads_generative` is the gradients from unsupervised training on the samples given in `opts.x_semisup`. The gradients are calculated with ´opts.classRBM=1`. We use samples of p(y I x_semisup) as labels. 

## Learning rate and momentum

The learning rate is controlled with `opts.learningrate`. `opts.learningrate` should be a handle to a function taking current epoch and momentum as input, this allows for decaying learning rate.

Decaying learning rate can be specified with:   

```MATLAB
eps       		  = 0.001;    % initial learning rate
f                 = 0.99;      % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);
```
Constant learning rate can be spcified with

```MATLAB
opts.learningrate = @(t,momentum) 0.01;
```

Momentum is controlled through `opts.momentum`. `opts.momentum` should be a function taking current epoch as input.

Ramp up momentum can be spcified with:

```MATLAB
T             = 50;       % momentum ramp up epoch
p_f 		  = 0.9;    % final momentum
p_i           = 0.5;    % initial momentum
opts.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);
```

Constant momentum can be specified with:

```MATLAB
opts.momentum = @(t) 0.9;
```


## weight initialization 

Initial weights are either sampled from a normal distriubtion [3] or from a uniform distribution [7], the behaivior is controlled thorugh `opts.init_type`:

```MATLAB
switch lower(opts.init_type)
    case 'gauss'
        initfunc = @(m,n) normrnd(0,0.01,m,n);
    case 'crbm'
        initfunc = @(m,n) (rand(m,n)-0.5) ./ max([m n]);
    otherwise
        error('init_type should be either gauss or cRBM');
end
```
Bias weights are always initialized to zero. 

## Regularization

The following regularization options are implemented

 * `opts.L1`: set the regularization weight
 * `opts.L2`: set the regularization weight
 * `opts.L2norm`: maximum L2 norm fo the incoming weights to each neuron [4]. Set the maximum L2norm for each neuron
 * `opts.sparsity`: implemented as in [7]. Specify the sparsity being subtracted after each epoch.
 * `opts.dropout_hidden`: dropout on hidden units. specify the probability of being dropped. see [1]
 * `opts.early_stopping`: Early stopping is available for classification RBM's where valdiation set is specified. The patience for early stopping can be set with `opts.patience`.


## Setting hiddenlayer sizes
In the example a RBM with 500 hidden units is created and trained. You do not need to set the size of the visible uints.

```MATLAB
sizes = [500];                        % hidden layer size
[opts, valid_fields] = dbncreateopts();   % create default opts struct
dbncheckopts(opts,valid_fields);          % simple check of validity of opts struct
dbn = dbnsetup(sizes, train_x, opts);     % create dbn struct
dbn = dbntrain(dbn, train_x, opts);       % train  dbn
```

You can stack several RBM by specifying sizes as a vector. `sizes = [500 200]` will stack two RBM's where the first RBM has 
#features visble units and 500 hidden units and the second RBM has 500 visble units and 200 hidden units. 
Any number of RBM's is allowed. If several RBM's are stacked non-top layer RBM's will be trained with generative training objective and `opts.classRBM = 0`. 

## Settings table

The table shows which fields in the opts struct that applies to the different training objectives.

|Setting   					| @genrative  	| @discriminative  	| @rbmhybrid  	| @rbmsemisublearn  	
|---						|---			|---				|---		 	|---					|
|init_type               	|  x 			|   x				|   	x		|  x 					|
|traintype   				|  x 			|	   				|   	x		|  x 					|
|cdn   						|  x 			|   				|   	x		|  x 					|
|numepochs   				|  x 			|   x				|   	x		|  x 					| 
]classRBM   				|  x 			|   x				|   	x		|  x 					| 
|err_func<sub>1</sub>   	|  x 			|   x				|   	x		|  x 					|
|test_interval<sub>1</sub> 	|  x 			|   x				|   	x		|  x 					|				
|learningrate   			|  x 			|   x				|   	x		|  x 					| 
|momentum   				|  x 			|   x				|   	x		|  x 					| 
|L1							|  x 			|   x				|   	x		|  x 					| 
|L2norm   					|  x 			|   x				|   	x		|  x 					| 
|sparsity   				|  x 			|   x				|   	x		|  x 					| 
|dropout_hidden   			|  x 			|   x				|   	x		|  x 					| 
|early_stopping<sub>1</sub> |  x 			|   x				|   	x		|  x 					| 
|patience<sub>1</sub>   	|  x 			|   x				|   	x		|  x 					| 
|y_train<sub>2</sub>   		|  x 			|   x				|   	x		|  x 					| 
|x_val<sub>2</sub>    		|  x 			|   x				|   	x		|  x 					| 		
|y_val<sub>2</sub>   		|  x 			|   x				|   	x		|  x 					| 
|x_semisup   				|   			|   				|   			|  x 					|
|hybrid_alpha   			|   			|   				|   	x		|   					|
|semisup_type   			|   			|   				|   			|  x 					|
|semisup_beta   			|   			|   				|   			|  x 					|

1) Applies if classRBM is 1 and x_val and y_val are set

2) Applies if classRBM is 1 

# Examples

## Example 1 - generative learning **p(x)**

Training RBM's in RBM_toolbox is controlled through three functions:
  * `dbncreateopts` creates an opts struct. The opts struct control learningrate, number of epochs, reqularization, training type etc. The help for `dbncreateopts` descripes all valid fields in the opts struct.
  * `dbnsetup` setups the DBN network, a single layer RBM is equal to a DBN. 
  * `dbntrain` trains the DBN

The following example trains a generative RBM with 500 hidden units and visulizes the found weights. Note that the learning rate is controlled through the `opts.learningrate` parameters. `opts.learningrate` is a function which takes the current epoch and current epoch as arguments and returns the learning rate. Similary  `opts.momentum` is a function that controls the current momentum. When the `opts.train_func` is set to `@rbmgenerative` RBM outputs the reconstruction error after each epoch, the reconstruction error should not be interpreted as a measure of goodness of the model, see [3].

```MATLAB
rng('default');rng(0);
load mnist_uint8;
train_x = double(train_x)/255;
train_y = double(train_y);


sizes = [200];   % hidden layer size
[opts, valid_fields] = dbncreateopts();

opts.numepochs = 30;
opts.traintype = 'CD';
opts.classRBM = 0;
opts.y_train = train_y;
opts.test_interval = 1;
opts.train_func = @rbmgenerative;
opts.init_type = 'cRBM';

opts.learningrate = @(t,momentum) 0.05;
opts.momentum     = @(t) 0;

dbncheckopts(opts,valid_fields);       
disp(opts)
dbn = dbnsetup(sizes, train_x, opts); 
dbn = dbntrain(dbn, train_x, opts);

% visualize weights
figure;visualize(dbn.rbm{1}.W(1:144,:)'); 
set(gca,'visible','off');
```

Visualization of weights:

<html>
<img src="/uploads/example1_weights.png" height="500" width="500"> 

## Example 2 - Generative RBM with labels **p(x,y)**

A classification RBM can be trained by setting `opts.classRBM` to 1 and and setting `opts.y_train` to the training labels. The training labels must be *one-of-K* encoded.

When `opts.classRBM` is 1 RBM toolbox will report the training error. The default error measure is accuracy but you may supply custom error measures through `opts.error_func`. If `opts.x_val` and `opts.y_val` are given the validation error will also be reported.
In the example the validation error is calculated after each epoch, i.e `opts.test_interval` is set 1. In the example we also enable early stopping, we use a early_stopping patience of 5, i.e if no progress have been made in 5 epochs stop training.

```MATLAB
rng('default');rng(0);
load mnist_uint8;
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y = double(test_y);

sizes = [200];  
[opts, valid_fields] = dbncreateopts();
opts.early_stopping = 1;
opts.patience = 5;

opts.numepochs = 50;
opts.traintype = 'CD';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = test_x;
opts.y_val = test_y;
opts.test_interval = 1;
opts.train_func = @rbmgenerative;
opts.init_type = 'cRBM';

opts.learningrate = @(t,momentum) 0.05;
opts.momentum     = @(t) 0;

dbncheckopts(opts,valid_fields);       
disp(opts)
dbn = dbnsetup(sizes, train_x, opts);  
dbn = dbntrain(dbn, train_x, opts);

% Make predictions
pred_val = dbnpredict(dbn,test_x);
[~, labels_val] = max(test_y,[],2);
acc_val = mean(pred_val == labels_val);
err_val = 1-acc_val

% plot weights 
figure;visualize(dbn.rbm{1}.W(1:144,:)'); 
set(gca,'visible','off');


% plot errors
plot([dbn.rbm{1}.val_error',dbn.rbm{1}.train_error'])
legend({'Validation error','Train error'})
[min_val,min_idx] =  min(dbn.rbm{1}.val_error);
hold on; plot(min_idx,min_val,'xr'); hold off;
xlabel('Epoch'); ylabel('Error'); grid on;
```

For classification RBM's predictions can be calculated by `dbnpredict` wich returns a label or with 
`dbnclassprobs` wich returns the predicted class probabilities. 

The learned weights can be visualized with the `visualize` function. 

<html>
<img src="/uploads/example2_weights.png" height="500" width="500"> 

The training erorror and validation error can be visualized as well:

<html>
<img src="/uploads/example2_error.png" height="350" width="350"> 

Note that in this example the validation error is lower than the training error, this is not typical. In the plot the 
red x indicate the lowest validation error. 

## Example 3 - PCD, layers and sampling

In example 3 we use PCD to train a classification DBN using the generative training objective. In the other examples `opts.traintype` has ben `CD` wich mean contrastive divergence [5]. In this example we will use `PCD`, persistent contrastive divergence [6].
In CD the gibbs chains are initiated at the data points, PCD differs from this by having a number of persistent chains wich are used to initiate the gibbs sampling.

```
clear all;
rng('default');rng(0);
load mnist_uint8;
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y = double(test_y);

sizes = [200 ];   % hidden layer size
[opts, valid_fields] = dbncreateopts();
opts.early_stopping = 1;
opts.patience = 5;
opts.numepochs = 50;
opts.traintype = 'PCD';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = test_x;
opts.y_val = test_y;
opts.test_interval = 1;
opts.train_func = @rbmgenerative;

%% Set learningrate
eps       		  = 0.05;    % initial learning rate
f                 = 0.95;      % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);

% Set momentum
T             = 50;       % momentum ramp up
p_f 		  = 0.9;    % final momentum
p_i           = 0.5;    % initial momentum
opts.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);


dbncheckopts(opts,valid_fields);       %checks for validity of opts struct
dbn = dbnsetup(sizes, train_x, opts);  % train function 
dbn = dbntrain(dbn, train_x, opts);

% plot weights 
figure;visualize(dbn.rbm{1}.W(1:144,:)'); 
set(gca,'visible','off');

% sample digits
class_vec = zeros(100,size(train_y,2));
for i = 1:size(train_y,2)
    class_vec((i-1)*10+1:i*10,i) = 1;
end

digits = dbnsample(dbn,100,10000,class_vec); 
figure;visualize(digits'); 
set(gca,'visible','off');

% sampling movie
dbnsamplemovie(dbn,10,3000,'example3',10,@visualize,eye(10))
```

Weight visualization:
<html>
<img src="/uploads/example3_weights.png" height="500" width="500"> 


`dbnsample` can sample from the model. The final sample is visualized below, with 

Samples visualization:
<html>
<img src="/uploads/example3_digits.png" height="500" width="500"> 


`dbnsamplemovie` can be used to create a movie of the sampling process as the gibbs chains converge, somehow similar to http://www.cs.toronto.edu/~hinton/adi/index.htm . 

[link to video](https://www.youtube.com/watch?v=qqdMu09_zm4) 



## Example 4 - Discriminative training

Look in folder  mnist_cRBM_discriminative 

## Example 5 - Hybrid training

## Example 6 - Semi-supervised learning 

## Example 7 - reproduce results from [7]

## References

[1] N. Srivastava and G. Hinton, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” J. Mach.  …, 2014.  
[2] H. Larochelle and Y. Bengio, “Classification using discriminative restricted Boltzmann machines,” … 25th Int. Conf. Mach. …, 2008.  
[3] G. Hinton, “A practical guide to training restricted Boltzmann machines,” Momentum, 2010.  
[4] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov, “Improving neural networks by preventing co-adaptation of feature detectors,” Jul. 2012.  
[5] G. Hinton, “Training products of experts by minimizing contrastive divergence,” Neural Comput., 2002.  
[6] T. Tieleman, “Training restricted Boltzmann machines using approximations to the likelihood gradient,” … 25th Int. Conf. Mach. …, 2008.  
[7] H. Larochelle and M. Mandel, “Learning algorithms for the classification restricted boltzmann machine,” J. Mach.  …, 2012.

Copyright (c) 2014, Søren Kaae Sønderby (skaaesonderby@gmail.com) All rights reserved.