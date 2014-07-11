if ~ismac
cd('/zhome/f9/4/69552/DeepLearnToolbox_noGPU')
addpath(genpath('/zhome/f9/4/69552/DeepLearnToolbox_noGPU'))
thrds  = str2num(getenv('PBS_NUM_PPN'));
else
thrds = 2;
end

disp(thrds)

parsave = @(fname) save(fname);



%set up a deepbelief network
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);



% create experiments
l = [1,0.1,0.01,00.1,0.001,0.0001];
train_type = [0,1];
experiments = combvec(l,train_type)';

all_lr = experiments(1,:);
all_tt = experiments(2,:);

n_experiments = size(experiments,1);


matlabpool('open',thrds);
parfor t = 1:n_experiments

%extract this experiments learningrate and training type
lr = all_lr(t);     
traintype = ifelse(all_tt(t) == 1, 'CD','PCD');

rand('state',0)
dbn.sizes = [500];
opts = dbncreateopts();
opts.traintype = traintype 
opts.numepochs =   100;
opts.batchsize = 100;
opts.cdn = 1; % contrastive divergence


T = 50;       % momentum ramp up
p_f = 0.9;    % final momentum
p_i = 0.5;    % initial momentum
eps = 0.01;    % initial learning rate
f = 0.9;     % learning rate decay

opts.learningrate = @(t,momentum) lr;
opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f, p_f);

opts.L1 = 0.00;
opts.L2 = 0;
opts.L2norm = 0;

opts.hintonDBN = 0;
opts.y_train = train_y;
opts.x_val = test_x;
opts.y_val = test_y;
opts.ratio_interval = 1;

dbn = dbnsetup(dbn, train_x, opts);

disp(opts)
disp(dbn)


dbn = dbntrain(dbn, train_x, opts);

%load('../dataanalysis-master/dbn_PCD_hintonDBN_nothinton.mat')
%figure; visualize(dbn.rbm{1}.W'); 

%digits = dbnsample(dbn,100,10000,5);
%figure; visualize(digits');
hinton = ifelse(opts.hintonDBN == 1,'hintonDBN','notHintonDBN')
outstr = ['dbn_singlelayer_' opts.traintype '_lr' strrep(num2str(lr),'.','-')...
'_' hinton '.mat']

parsave(outstr);
end
matlabpool('close');


