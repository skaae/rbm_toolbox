# RBM Toolbox

Toolbox for training RBM's and DBN's. 
Support for joint training of features and labels. 

Significiant additions:
 * add support for training RBM's with class labels 
 * CD - k (contrastive divergence k)
 * PCD (persistent contrastive divergence)
 * Various rbm sampling functions (pictures / movies)
 * Classiciation support
 * Regularization: L1, L2, sparsity
 * discriminative RBM training (work in progress)

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
	preds = dbnpredict(dbn,train_x);


This toolbox builds on the DeepLearnToolbox by Rasmus Berg Palm.


 Copyright (c) 2014, Søren Kaae Sønderby (skaaesonderby@gmail.com)
All rights reserved.