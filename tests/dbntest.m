function dbntest()
% test generative and discrminative weights
dbn_cRBM_tests_gpu()

% test that everything runs for a few epochs
[train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist(101,0.001);
sizes = [11];
[opts] = dbncreateopts();
opts.gpu = -1;
opts.testinterval = 1;
opts.y_train = train_y;

% test single epoch
opts.numepochs = 1;
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);


opts.numepochs = 2;
%% Test without validation set
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% Test with validationset 
opts.x_val = val_x;
opts.y_val = val_y;

%% discrminative
opts.beta = 0;
opts.alpha = 0; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% generative
opts.beta = 0;
opts.alpha = 1; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% hybrid
opts.beta = 0;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);


opts.x_semisup = test_x;
%% semisup  + discriminative
opts.beta = 0.5;
opts.alpha = 0; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% semisup  + generative
opts.beta = 0.5;
opts.alpha = 1; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% semisup  + hybrid
opts.beta = 0.5;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);


disp('ALL PASSED!')
end
