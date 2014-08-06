function dbntest()
% test generative and discrminative weights
dbn_cRBM_tests()

% test that everything runs for a few epochs
[train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist(101,0.001);
sizes = [11];
[opts, valid_fields] = dbncreateopts();
opts.test_interval = 1;
opts.batchsize = 1;
opts.numepochs = 2;

%% check generative with only x given not class RBM
opts.classRBM = 0;
opts.train_func = @rbmgenerative;
dbncheckopts(opts,valid_fields);
dbn1 = dbnsetup(sizes, train_x, opts);
dbn1 = dbntrain(dbn1, train_x, opts);

%% generative with x and validation_x
opts.x_val = val_x;
dbncheckopts(opts,valid_fields);
dbn2 = dbnsetup(sizes, train_x, opts);
dbn2 = dbntrain(dbn2, train_x, opts);

%% generative with x and y no validation, classRBM
opts.x_val = [];
opts.classRBM = 1;
opts.y_train = train_y;
dbncheckopts(opts,valid_fields);
dbn3 = dbnsetup(sizes, train_x, opts);
dbn3 = dbntrain(dbn3, train_x, opts);

%% generative with x and y + validation, classRBM
opts.x_val = val_x;
opts.y_val = val_y;
dbncheckopts(opts,valid_fields);
dbn4 = dbnsetup(sizes, train_x, opts);
dbn4 = dbntrain(dbn4, train_x, opts);

%% rbmdiscrminative
opts.train_func =  @rbmdiscriminative;
dbncheckopts(opts,valid_fields);
dbn5 = dbnsetup(sizes, train_x, opts);
dbn5 = dbntrain(dbn5, train_x, opts);

%% hybrid
opts.train_func =  @rbmhybrid;
dbncheckopts(opts,valid_fields);
dbn6 = dbnsetup(sizes, train_x, opts);
dbn6 = dbntrain(dbn6, train_x, opts);


%% semisuplearn
opts.train_func =  @rbmsemisuplearn;
opts.x_semisup = test_x;
dbncheckopts(opts,valid_fields);
dbn7 = dbnsetup(sizes, train_x, opts);
dbn7 = dbntrain(dbn7, train_x, opts);



% check all types of training with larger batch size
opts.batchsize = 10;
opts.train_func =  @rbmgenerative;
dbn8 = dbnsetup(sizes, train_x, opts);
dbn8 = dbntrain(dbn8, train_x, opts);

opts.train_func =  @rbmdiscriminative;
dbn9 = dbnsetup(sizes, train_x, opts);
dbn9 = dbntrain(dbn9, train_x, opts);

opts.train_func =  @rbmdiscriminative;
dbn10 = dbnsetup(sizes, train_x, opts);
dbn10 = dbntrain(dbn10, train_x, opts);

opts.train_func =  @rbmdiscriminative;
dbn11 = dbnsetup(sizes, train_x, opts);
dbn11 = dbntrain(dbn11, train_x, opts);


%% check early stopping
opts.early_stopping = 1;
opts.train_func =  @rbmgenerative;
opts.outfile = 'test.mat';
dbn12 = dbnsetup(sizes, train_x, opts);
dbn12 = dbntrain(dbn12, train_x, opts);

%% check regularization
opts.L1 = 0.1;
opts.L2 = 0.1;
opts.L2norm = 0.1;
opts.sparsity = 0.1;
opts.train_func =  @rbmgenerative;
dbn13 = dbnsetup(sizes, train_x, opts);
dbn13 = dbntrain(dbn13, train_x, opts);

%% check init types
opts.init_type = 'gauss';
dbnsetup(sizes, train_x, opts);

opts.init_type = 'crbm';
dbnsetup(sizes, train_x, opts);


disp('ALL PASSED!')
end
