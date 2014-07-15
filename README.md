# RBM Toolbox

Toolbox for training RBM's and DBN's. 
Support for joint training of features and labels. 

Significiant additions:
 * add support for training RBM's with class labels including, see [1,2]
    * generative training objective
    * discriminative training objective
    * hybrid training objective
 * CD - k (contrastive divergence k)
 * PCD (persistent contrastive divergence)
 * Various rbm sampling functions (pictures / movies)
 * Classiciation support
 * Regularization: L1, L2, maxL2norm, sparsity, early-stopping
 * Support for custom error functions

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

 Copyright (c) 2014, Søren Kaae Sønderby (skaaesonderby@gmail.com)
All rights reserved.