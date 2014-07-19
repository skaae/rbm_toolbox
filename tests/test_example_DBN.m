function test_example_DBN
load mnist_uint8;

 train_x = double(train_x(1:30000,:)) / 255;
% val_x   = train_x(1:1000,:);
% semisup_x = train_x(1001:2000,:);
% train_x(1:59000,:) = [];
% 
% train_y = double(train_y);
% val_y   = train_y(1:1000,:);
% train_y(1:59000,:) = [];
% 
% test_x  = double(test_x(1:1000))  / 255;
% test_y  = double(test_y(1:1000));





%%  Generative RBM with 100 hidden units and contrastive divergence
% note that the learningrate and the momentum must be specified as functions.
% This allows for decaying learningrate and momentum rampup.
% The example uses momentum with rampup and decaying learning rate

rng('default');rng(1);
sizes = [500];

[opts, valid_fields] = dbncreateopts();
opts.numepochs = 50;
opts.traintype = 'CD'
opts.train_func = @rbmgenerative;

T = 25;       % momentum ramp up
p_f = 0.9;    % final momentum
p_i = 0.5;    % initial momentum
opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);

eps = 0.05;    % initial learning rate
f = 0.97;     % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);

%opts.momentum = @(t) 0;
dbncheckopts(opts,valid_fields);
dbn = dbnsetup(sizes, train_x, opts);
opts.sparsity = 0.1;  
dbn1 = dbntrain(dbn, train_x, opts);
figure;visualize(dbn1.rbm{1}.W');

opts.sparsity = 0.01;  
dbn2 = dbntrain(dbn, train_x, opts);
figure;visualize(dbn2.rbm{1}.W');

opts.sparsity = 0.000;  
dbn3 = dbntrain(dbn, train_x, opts);
figure;visualize(dbn3.rbm{1}.W(1:144,:)');

opts.sparsity = 0.0001;  
dbn4 = dbntrain(dbn, train_x, opts);
figure;visualize(dbn4.rbm{1}.W');


%%
dbncheckopts(opts,valid_fields);
dbn = dbnsetup(sizes, train_x, opts);
opts.L1 = 0.1;  
dbn5 = dbntrain(dbn, train_x, opts);
figure;visualize(dbn5.rbm{1}.W');

opts.L1 = 0.01;  
dbn6 = dbntrain(dbn, train_x, opts);
figure;visualize(dbn6.rbm{1}.W');

opts.L1 = 0.001;  
dbn7 = dbntrain(dbn, train_x, opts);
figure;visualize(dbn7.rbm{1}.W');

opts.L1 = 0.0001;  
dbn8 = dbntrain(dbn, train_x, opts);
figure;visualize(dbn8.rbm{1}.W');


% visualize learning rate and momentum
l = []; m = [];
for i = 1:50
    m(end+1) = opts.momentum(i);
    l(end+1) = opts.learningrate(i,m(i));  
end
[AX,H1,H2] = plotyy(1:50,m,1:50,l,'plot');
set(get(AX(1),'Ylabel'),'String','Momentum') 
set(get(AX(2),'Ylabel'),'String','Learning rate') 
set(get(AX(1),'Xlabel'),'String','Epoch') 


%%


dbn.sizes = [100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   100;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
