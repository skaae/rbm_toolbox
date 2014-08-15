function dbntest()
% test generative and discrminative weights
dbn_cRBM_tests_gpu()

% test that everything runs for a few epochs
[train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist(0.001);
sizes = [7];
[opts] = dbncreateopts();
opts.gpu = -1;
opts.testinterval = 1;
opts.y_train = train_y(:,1:10);

% test single epoch
opts.numepochs = 1;
dbn_no_gpu = dbnsetup(sizes, train_x(), opts);
dbntrain(dbn_no_gpu, train_x, opts);

% test cdn larger than 1
opts.cdn = 3;
opts.beta = 0;
opts.alpha = 1;
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

% test traintestbatch not []
opts.beta = 0;
opts.alpha = 1;
opts.traintestbatch = 3;
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);
opts.traintestbatch = []; % default

% test regularization
opts.cdn = 1;
opts.beta = 0;
opts.alpha = 1;
opts.L1 = 0.1;
opts.L2 = 0.1;
opts.sparsity = 0.1;
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

% test dropout
opts.dropout = 0.5;
opts.alpha = 0.5
dbn_no_gpu = dbnsetup([50], train_x(:,1:50), opts);
dbntrain(dbn_no_gpu, train_x(:,1:50), opts);
opts.alha = 1;
% test dropconnect
opts.dropout = 0;
opts.dropconnect = 0.5;
dbn_no_gpu = dbnsetup([50], train_x(:,1:50), opts);
dbntrain(dbn_no_gpu, train_x(:,1:50), opts);



%% test two epochs
opts.traintype = 'CD';
opts.cdn = 1;
opts.numepochs = 2;
opts.dropconnect = 0;

%% Test without validation set
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% Test with validationset 
opts.x_val = val_x;
opts.y_val = val_y;

%% discrminative
opts.beta = 0;
opts.alpha = 0; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% generative
opts.beta = 0;
opts.alpha = 1; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% hybrid
opts.beta = 0;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% hybrid two layers
opts.beta = 0;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup([11 11], train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% hybrid + PCD
opts.traintype = 'PCD';
opts.npcdchains = 5;
opts.beta = 0;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);


opts.traintype = 'CD';
opts.x_semisup = test_x;
%% semisup  + discriminative
opts.beta = 0.5;
opts.alpha = 0; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% semisup  + generative
opts.beta = 0.5;
opts.alpha = 1; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% semisup  + hybrid
opts.beta = 0.5;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% semisup  + hybrid + two layers
opts.beta = 0.5;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup([11 13], train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

%% semisup  + hybrid + PCD
opts.traintype = 'PCD';
opts.beta = 0.5;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
dbntrain(dbn_no_gpu, train_x, opts);

disp('ALL PASSED!')
end
