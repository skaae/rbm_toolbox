[train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist();
sizes = [200];
[opts, valid_fields] = dbncreateopts();

opts.y_train = train_y;
opts.numepochs = 5;
dbncheckopts(opts,valid_fields);


dbn = dbnsetup(sizes, train_x, opts);

opts.train_func = @rbmdiscriminative;
rng('default');rng(101);
profile on; profile clear
dbn = dbntrain(dbn, test_x, opts);
profile off
profile viewer