# RBM Toolbox

RBM toolbox is a MATLAB toolbox for training RBM's. It builds on the DeepLearnToolbox by Rasmus Berg.

RBM toolbox support among others:

 * add support for training RBM's with class labels including, see [1,2]
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

## training objectives

The RBM toolbox supports four different TBM training objectives. For a detailed description refer to [2].


* `rbmgenerative`:  `-log(p(x))` or `-log(p(x,y)) if `classRBM` is 1
* `rbmdiscriminative`  `-log(p(x I y))`   [2]
* `rbmhybrid` Models   `-(1-alpha)log(y I x) - alpha log(p (x) )` [2]
* `rbmsemisuplearn`    `TYPE + unsupervised. Where type is {generative, discriminative,hybrid} and unsupervised is generative training on unlabeled data  [2]


The RBM training objective is set by supplying a function handle to one of the four training functions through `opts.train_func` 

## Regularization

RBM toolbox supports L1 and L2 regularization and regularization through a maximum L2 norm fo the incoming weights to each neuron [4].
Sparsity is implemented as described in [2]. Dropout of hidden units is implemented as described in [1].

When training a classification RBM ('opts.classRBM = 1') and a validation set is given through `opts.x_val` and `opts.y_val`, then early stopping can be used. The patience for early stopping can be specified with `opts.patience`. 

## Training DBN's
DBN can be trained by given multiple hidden sizes to `dbnsetup` e.g. `sizes=[500 500]` for a two layer DBN with 500 hidden units in 
each layer. 

**TODO** add wake sleep algorithm?

## Settings table

The table shows which fields in the opts struct that applies to the different training objectives.

|Setting   					| @genrative  	| @discriminative  	| @rbmhybrid  	| @rbmsemisublearn  	
|---						|---			|---				|---		 	|---					|
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
train_x = double(train_x) / 255;

sizes = [500];   % hidden layer size
[opts, valid_fields] = dbncreateopts();
opts.numepochs = 50;
opts.traintype = 'CD';
opts.classRBM = 0;
opts.train_func = @rbmgenerative;


%% Set learningrate
eps       		  = 0.05;    % initial learning rate
f                 = 0.97;      % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);

% Set momentum
T             = 25;       % momentum ramp up
p_f 		  = 0.9;    % final momentum
p_i           = 0.5;    % initial momentum
opts.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);


dbncheckopts(opts,valid_fields);       %checks for validity of opts struct
dbn = dbnsetup(sizes, train_x, opts);  % train function 
dbn = dbntrain(dbn, train_x, opts);
figure;
figure;visualize(dbn.rbm{1}.W(1:144,:)'); 
set(gca,'visible','off');
```

In the example the learningrate (blue) starts at *0.05* and decays with each epoch. The momentum (green) ramps up over 25 epochs, as shown in the figure. 

<html>
<img src="/uploads/learnmom.png" height="350" width="350"> 

Finally the weights can be visualized:

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

sizes = [500];   % hidden layer size
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

%% Set learningrate
eps       		  = 0.05;    % initial learning rate
f                 = 0.97;      % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);

% Set momentum
T             = 25;       % momentum ramp up
p_f 		  = 0.9;    % final momentum
p_i           = 0.5;    % initial momentum
opts.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);

dbncheckopts(opts,valid_fields);       %checks for validity of opts struct
dbn = dbnsetup(sizes, train_x, opts);  % train function 
dbn = dbntrain(dbn, train_x, opts);

%% Do predictions
pred_val = dbnpredict(dbn,test_x);
[~, labels_val] = max(test_y,[],2);
acc_val = mean(pred_val == labels_val);

%% plot weights 
figure;visualize(dbn.rbm{1}.W(1:144,:)'); 
set(gca,'visible','off');

%% plot errors
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

## Example 3 - PCD sampling and movies

## Example 4 - Discriminative training

## Example 5 - Hybrid training

## Example 6 - Semi-supervised learning 

## References

[1] N. Srivastava and G. Hinton, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” J. Mach.  …, 2014.  
[2] H. Larochelle and Y. Bengio, “Classification using discriminative restricted Boltzmann machines,” … 25th Int. Conf. Mach. …, 2008.  
[3] G. Hinton, “A practical guide to training restricted Boltzmann machines,” Momentum, 2010.  
[4] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. R. Salakhutdinov, “Improving neural networks by preventing co-adaptation of feature detectors,” Jul. 2012.  

Copyright (c) 2014, Søren Kaae Sønderby (skaaesonderby@gmail.com) All rights reserved.