[train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist(0,0.1);
sizes = [200];
[opts, valid_fields] = dbncreateopts();

opts.y_train = train_y;
opts.init_type = 'cRBM';
opts.numepochs = 1;
opts.test_interval = 1;
opts.train_func = @rbmdiscriminative;
opts.classRBM = 1;
opts.batchsize = 1;
dbncheckopts(opts,valid_fields);


dbn = dbnsetup(sizes, train_x, opts);



rng('default');rng(101);

close all; profile on; profile clear
tic
dbn = dbntrain(dbn, train_x, opts);
toc
profile off
profile viewer


% testing sampling matrix
% s = rand(20,10);
% s = bsxfun(@rdivide,s,sum(s,2));  % normalize each row
% 
% profile on; profile clear
% for i = 1:1000
% rand('state',i);
% s1 = samplematrix(s);
% rand('state',i);
% s2 = samplematrixtest(s);
% assert(isequal(s1,s2))
% end
% 
% profile off
% profile viewer