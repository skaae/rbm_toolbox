if ~ismac
    cd('/zhome/f9/4/69552/DeepLearnToolbox_noGPU')
    addpath(genpath('/zhome/f9/4/69552/DeepLearnToolbox_noGPU'))
end
%set up a deepbelief network
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN

% for testing
% fprintf('TESTING ON\n')
% train_x = train_x(1:100,:);
% train_y = train_y(1:100,:);
% test_x = test_x(1:100,:);
% test_y = test_y(1:100,:);

rand('state',0)
dbn.sizes = [500];
opts = dbncreateopts();
opts.traintype = 'PCD';
opts.numepochs =   500;
opts.batchsize = 100;
opts.cdn = 1; % contrastive divergence

T = 50;       % momentum ramp up
p_f = 0.9;    % final momentum
p_i = 0.5;    % initial momentum
eps = 0.01;    % initial learning rate
f = 0.9;     % learning rate decay
%opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum); 
 opts.learningrate = @(t,momentrum) 0.0001;
opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);
%opts.momentum     = @(t) 0;
opts.L1 = 0.00;
opts.L2 = 0;
opts.L2norm = 0;

opts.hintonDBN = 1;
opts.y_train = train_y;
opts.x_val = test_x;
opts.y_val = test_y;

dbn = dbnsetup(dbn, train_x, opts);

disp(opts)
disp(dbn)


dbn = dbntrain(dbn, train_x, opts);

%load('../dataanalysis-master/dbn_PCD_hintonDBN.mat')
%figure; visualize(dbn.rbm{1}.W'); 

%digits = dbnsample(dbn,100,10000,5);
%figure; visualize(digits');

save('dbn_singlelayer_PCD_small_learningrate.mat');
