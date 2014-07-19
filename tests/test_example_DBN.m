function test_example_DBN
load mnist_uint8;

train_x = double(train_x)/255;
%train_x = double(train_x(1:30000,:)) / 255;
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
 
dbn = dbntrain(dbn, train_x, opts);
figure;visualize(dbn.rbm{1}.W(1:144,:)');
set(gca,'visible','off');

%% example 2
rng('default');rng(0);
load mnist_uint8;
train_x = double(train_x)/255;
vx   = double(train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);


sizes = [500];   % hidden layer size
[opts, valid_fields] = dbncreateopts();
opts.numepochs = 50;
opts.traintype = 'CD';
opts.classRBM = 1;
opts.y_train = ty;
opts.x_val = vx;
opts.y_val = vy;
opts.train_func = @rbmgenerative;


%% Set learningrate
eps       		  = 0.05;    % initial learning rate
f                 = 0.97;      % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);

% Set momentum
T             = 25;       % momentum ramp up
p_f 		  = 0.9;    % final momentum
p_i           = 0.5;    % initial momentum
opts.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);


dbncheckopts(opts,valid_fields);       %checks for validity of opts struct
dbn = dbnsetup(sizes, tx, opts);  % train function 
dbn = dbntrain(dbn, tx, opts);
figure;
figure;visualize(dbn.rbm{1}.W(1:144,:)'); 
set(gca,'visible','off');


