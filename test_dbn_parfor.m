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
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);



% create experiments
l = 5*10.^(-[2 3 4 5]);
train_type = [0 1];
hin        = 1;
l1 = [0 0.0001];
n_hid = [100, 500];
experiments = combvec(l,train_type,hin,l1,n_hid)';

all_lr = experiments(:,1);
all_tt = experiments(:,2);
all_hin = experiments(:,3);
all_l1 = experiments(:,4);
all_hid = experiments(:,5);

n_experiments = size(experiments,1);


all_dbn = {};
all_opts = {};
for t = 1:n_experiments
    %extract this experiments learningrate and training type
    lr = all_lr(t);
    traintype = ifelse(all_tt(t) == 1, 'CD','PCD');
    hinton = all_hin(t);
    
    dbn.sizes = [all_hid(t)];
    opts = dbncreateopts();
    opts.traintype = traintype;
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
    
    opts.L1 = all_l1(t);
    opts.L2 = 0;
    opts.L2norm = 0;
    
    opts.classRBM = hinton;
    opts.y_train = train_y;
    opts.x_val = test_x;
    opts.y_val = test_y;
    opts.test_interval = 1;
    
    hinton = ifelse(opts.classRBM == 1,'classRBM','notclassRBM');
    outstr = ['rbm_' opts.traintype '_lr' num2str(lr)...
        '_L1' num2str(opts.L1) '_' hinton '-' num2str(all_hid(t))];
    outstr = strrep(outstr,'.','-');
    opts.outstr = outstr;
    dbn = dbnsetup(dbn, train_x, opts);
    
    if exist([opts.outstr '.mat'], 'file') == 2
        disp(['experiment already calculated: ' opts.outstr]);
    else
        disp(['add experiment: ' opts.outstr]);
        all_dbn{end+1} = dbn;
        all_opts{end+1} = opts;
    end
    clear opts dbn
    
end

class_vec = zeros(100,size(train_y,2));
for i = 1:size(train_y,2)
    class_vec((i-1)*10+1:i*10,i) = 1;
end

rand('state',0);




parfor t = 1:numel(all_dbn)
    opts = all_opts{t};
    dbn  = all_dbn{t};
    dbn = dbntrain(dbn, train_x, opts);
    digits = dbnsample(dbn,100,10000,class_vec);
    parsave(opts.outstr,dbn,opts,digits);
end
matlabpool('close');


