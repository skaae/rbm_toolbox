if ~ismac
    cd('/zhome/f9/4/69552/DeepLearnToolbox_noGPU')
    addpath(genpath('/zhome/f9/4/69552/DeepLearnToolbox_noGPU'))
    try
        thrds  = str2num(getenv('PBS_NUM_PPN'));
        matlabpool('open',thrds);
    catch e
        disp(e)
        thrds = 0;
    end
    
else
    thrds = 2;
end

disp(thrds)

parsave = @(fname,dbn,opts,digits) save(fname,'dbn','opts','digits');



%set up a deepbelief network
rand('state',0);
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);


%%% for testing
% train_x = train_x(1:100,:);
% test_x =  test_x(1:100,:);
% train_y = train_y(1:100,:);
% test_y =  test_y(1:100,:);
% [~, labs] = max(train_y,[],2);
% 
% A = prdataset(train_x,labs);
% prrbm =  drbmc(A, 100);


dbn.sizes = [500];
[opts valid_fields] = dbncreateopts();

opts.train_func = @rbmhybrid;

opts.traintype = 'PCD';
opts.numepochs =   100;
opts.batchsize = 100;
opts.cdn = 1; % contrastive divergence


T = 50;       % momentum ramp up
p_f = 0.9;    % final momentum
p_i = 0.5;    % initial momentum
eps = 0.01;    % initial learning rate
f = 0.9;     % learning rate decay

opts.learningrate = @(t,momentum) 0.005;
opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f, p_f);
%opts.momentum     = @(t) 0;
opts.L1 = 0;
opts.L2 = 0;
opts.L2norm = 0;

opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = test_x;
opts.y_val = test_y;
opts.test_interval = 1;
opts.early_stopping = 0;
opts.patience = 20;

dbncheckopts(opts,valid_fields);
dbn = dbnsetup(dbn, train_x, opts);

dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W')



