
% test generative and discrminative weights
dbn_cRBM_tests()

% test that everything runs for a few epochs
[train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist(101,0.001);
sizes = [11];
[opts, valid_fields] = dbncreateopts();
opts.test_interval = 1;

opts.batchsize = 1;
opts.numepochs = 2;

%%%% check generative with only x given not class RBM
opts.classRBM = 0;
opts.train_func = @rbmgenerative;
dbncheckopts(opts,valid_fields);
dbn1 = dbnsetup(sizes, train_x, opts);
dbn1 = dbntrain(dbn1, train_x, opts);







opts.y_train = train_y;





opts.x_val = val_x;
opts.y_val = val_y;
opts.init_type = 'cRBM';


