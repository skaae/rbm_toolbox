%% Example 4 - Discriminative training
% Tries to reproduce discriminative result from table 1 in 
% "Learning algorithms for the classification Restricted boltzmann machine"
if ~ismac
    cd('/zhome/ba/d/51602/rbm_toolbox')
    addpath(genpath('/zhome/ba/d/51602/rbm_toolbox'))

end

%% setup training
rng('default');rng(0);
load mnist_uint8;

% Test set
test_x  = double(test_x)/255;
test_y = double(test_y);

%Training and validation set
train_x = double(train_x)/255;
train_y = double(train_y);

val_x   = train_x(1:10000,:);
train_x = train_x(10001:end,:);
val_y   = train_y(1:10000,:);
train_y = train_y(10001:end,:);

% Setup DBN
sizes = [500 ];   % hidden layer size
[opts, valid_fields] = dbncreateopts();
opts.early_stopping = 1;
opts.patience = 15;
opts.numepochs = 10000;
opts.traintype = 'CD';
opts.init_type = 'cRBM';
opts.test_interval = 1;

opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = val_x;
opts.y_val = val_y;
opts.train_func = @rbmdiscriminative;

%% Set learningrate and momentum
opts.learningrate = @(t,momentum) 0.05;
opts.momentum = @(t) 0;

disp(opts)
dbncheckopts(opts,valid_fields);       %checks for validity of opts struct
dbn = dbnsetup(sizes, train_x, opts);  % train function 
dbn = dbntrain(dbn, train_x, opts);


class_vec = zeros(100,size(train_y,2));
for i = 1:size(train_y,2)
    class_vec((i-1)*10+1:i*10,i) = 1;
end
digits = dbnsample(dbn,100,10000,class_vec);


save('example.mat','dbn','opts','digits');
