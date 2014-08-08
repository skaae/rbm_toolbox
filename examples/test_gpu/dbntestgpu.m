function dbntestgpu()
% test generative and discrminative weights
% test that everything runs for a few epochs
if ~ismac
    current_dir = pwd();
    cd('../..');
    addpath(genpath(pwd()));
    cd(current_dir)
    
    getenv('ML_GPUDEVICE')
    gpuidx = str2num(getenv('ML_GPUDEVICE')) + 1

    gpu = gpuDevice(gpuidx);
    reset(gpu);
    wait(gpu);
    
    disp(['GPU memory available (Gb): ', num2str(gpu.FreeMemory / 10^9)]);
else 
    gpu = [];
end


[train_x,val_x,test_x,train_y,val_y,test_y] =setupmnist(101,0.001);
sizes = [11];
[opts] = dbncreateopts();
opts.gpu = 1;
opts.testinterval = 1;
opts.thisgpu = gpu;
opts.y_train = train_y;

% test single epoch
% test single epoch
opts.numepochs = 1;
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);


% test cdn larger than 1
opts.cdn = 3;
opts.beta = 0;
opts.alpha = 1;
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);


% test regularization
opts.cdn = 1;
opts.beta = 0;
opts.alpha = 1;
opts.L1 = 0.1;
opts.L2 = 0.1;
opts.sparsity = 0.1;
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% test two epochs
opts.traintype = 'CD';
opts.cdn = 1;
opts.numepochs = 2;
%% Test without validation set
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% Test with validationset 
opts.x_val = val_x;
opts.y_val = val_y;

%% discrminative
opts.beta = 0;
opts.alpha = 0; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% generative
opts.beta = 0;
opts.alpha = 1; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% hybrid
opts.beta = 0;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% hybrid + PCD
opts.traintype = 'PCD';
opts.npcdchains = 5;
opts.beta = 0;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);


opts.traintype = 'CD';
opts.x_semisup = test_x;
%% semisup  + discriminative
opts.beta = 0.5;
opts.alpha = 0; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% semisup  + generative
opts.beta = 0.5;
opts.alpha = 1; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% semisup  + hybrid
opts.beta = 0.5;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);

%% semisup  + hybrid + PCD
opts.traintype = 'PCD';
opts.beta = 0.5;
opts.alpha = 0.5; 
dbn_no_gpu = dbnsetup(sizes, train_x, opts);
rbm_no_gpu = dbn_no_gpu.rbm{1};
rbmtraingpu(rbm_no_gpu, train_x, opts);


disp('ALL PASSED!')
end
