
if ~ismac
    cd('/zhome/f9/4/69552/DeepLearnToolbox_noGPU')
    addpath(genpath('/zhome/f9/4/69552/DeepLearnToolbox_noGPU'))
    
    addpath('~/dataanalysis-master/')
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
semisup_x = [];


%%% for testing
% semisup_x = train_x(3001:end,:);
 %train_x = train_x(1:500,:);
 %test_x =  test_x(1:100,:);
 %train_y = train_y(1:500,:);
 %test_y =  test_y(1:100,:);



sizes = [200];
[opts, valid_fields] = dbncreateopts();

opts.train_func = @rbmgenerative;
opts.dropout_hidden = 0;

opts.traintype = 'CD';
opts.numepochs =   1000;
opts.batchsize = 100;
opts.cdn = 1; % contrastive divergence

T = 80;       % momentum ramp up
p_f = 0.9;    % final momentum
p_i = 0.5;    % initial momentum
eps = 0.01;    % initial learning rate
f = 0.9;     % learning rate decay

opts.learningrate = @(t,momentum) 0.01;
opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f, p_f);
opts.momentum     = @(t) 0;
opts.L1 = 0;
opts.L2 = 0;
opts.L2norm = 0;

opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = test_x;
opts.y_val = test_y;
opts.x_semisup = semisup_x;

opts.test_interval = 1;
opts.early_stopping = 1;
opts.patience = 10;
opts.hybrid_alpha = 0.5;
opts.err_func = @accuracy;

dbncheckopts(opts,valid_fields);
dbn = dbnsetup(sizes, train_x, opts);

dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W(1:100,:)')



