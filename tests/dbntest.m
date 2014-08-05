
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






opts.y_train = train_y;





opts.x_val = val_x;
opts.y_val = val_y;
opts.init_type = 'cRBM';


