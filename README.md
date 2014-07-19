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
|Setting   			| @genrative  	| @discriminative  	| @rbmhybrid  	| @rbmsemisublearn  	|
|---				|---			|---				|---		 	|---					|---
|traintype   		|   			|	   				|   			|   					|
|cdn   				|   			|   				|   			|   					|
|numepochs   		|  x 			|   x				|   	x		|  x 					| x
]classRBM   		|  x 			|   x				|   	x		|  x 					| x
|err_func $$_1$$   	|  x 			|   x				|   	x		|  x 					|
|test_interval $$_1$$|  x 			|   x				|   	x		|  x 					|				
|learningrate   				|   	|   	|   				|   	|
|momentum   				|   	|   	|   				|   	|
|L1			|   	|   	|   				|   	|
|L2norm   				|   	|   	|   				|   	|
|sparsity   				|   	|   	|   				|   	|
|dropout_hidden   				|   	|   	|   				|   	|
|early_stopping   				|   	|   	|   				|   	|
|patience   				|   	|   	|   				|   	|
|y_train   				|   	|   	|   				|   	|
|x_val   				|   	|   	|   				|   	|
|y_val  				|   	|   	|   				|   	|
|x_semisup   				|   	|   	|   				|   	|
|hybrid_alpha   				|   	|   	|   				|   	|
|semisup_type   				|   	|   	|   				|   	|
|semisup_beta   				|   	|   	|   				|   	|

1) Applies if classRBM is 1

# Examples

## Example 1 - generative training $$p(x)$$

Training RBM's in RBM_toolbox is controlled through three functions:
  * `dbncreateopts` creates an opts struct. The opts struct control learningrate, number of epochs, reqularization, training type etc. The help for `dbncreateopts` descripes all valid fields in the opts struct.
  * `dbnsetup` setups the DBN network, a single layer RBM is equal to a DBN. 
  * `dbntrain` trains the DBN

The following example trains a generative RBM with 500 hidden units and visulizes the found weights. Note that the learning rate is controlled through the `opts.learningrate` parameters. `opts.learningrate` is a function which takes the current epoch and current epoch as arguments and returns the learning rate. Similary  `opts.momentum` is a function that controls the current momentum. When the `opts.train_func` is set to `@rbmgenerative` RBM outputs the reconstruction error after each epoch, the reconstruction error should not be interpreted as a measure of goodness of the model, see [3].

```MATLAB
rng('default');rng(0);
load mnist_uint8;
 train_x = double(train_x(1:30000,:)) / 255;

sizes = [500];   % hidden layer size
[opts, valid_fields] = dbncreateopts();
opts.numepochs = 50;
opts.traintype = 'PCD';
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


For the 

 # Example usage


		%load Data
		rand('state',0);
		load mnist_uint8;

		train_x = double(train_x) / 255;
		test_x  = double(test_x)  / 255;
		train_y = double(train_y);
		test_y  = double(test_y);


		% for sampling from RBM
		class_vec = zeros(100,size(train_y,2));
		for i = 1:size(train_y,2)
		    class_vec((i-1)*10+1:i*10,i) = 1;
		end

		% SETUP and train
		dbn.sizes = [500];
		opts = dbncreateopts();
		dbn = dbntrain(dbn, train_x, opts);
		digits = dbnsample(dbn,100,5000,class_vec);
		dbnsamplemovie(dbn,100,3000,'test',1,@visualize,class_vec);
		probs = dbnclassprobs(dbn,train_x);
		preds = dbnpredict(dbn,train_x);


This toolbox builds on the DeepLearnToolbox by Rasmus Berg Palm.

## References

[1] N. Srivastava and G. Hinton, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” J. Mach.  …, 2014.
[2] H. Larochelle and Y. Bengio, “Classification using discriminative restricted Boltzmann machines,” … 25th Int. Conf. Mach. …, 2008.
[3] G. Hinton, “A practical guide to training restricted Boltzmann machines,” Momentum, 2010.
 Copyright (c) 2014, Søren Kaae Sønderby (skaaesonderby@gmail.com)
All rights reserved.