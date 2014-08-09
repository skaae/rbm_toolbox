%% Example 8 - Sparse hybrid trained on subset. Baseline for semisupervised 
% Training set is resized to 5000 samples and validation set to 1000 samples.
% The test set is not resized.
if ~ismac
    current_dir = pwd();
    cd('../..');
    addpath(genpath(pwd()));
    cd(current_dir)
    
    getenv('ML_GPUDEVICE')
    gpuidx = str2num(getenv('ML_GPUDEVICE')) + 1

    gpu = gpuDevice(gpuidx);
    disp(gpu);
    reset(gpu);
    wait(gpu);
    
    disp(['GPU memory available (Gb): ', num2str(gpu.FreeMemory / 10^9)]);
else 
    gpu = [];
end

downsize = 0.1;


name = 'example_9_semisup100';
rng('default');rng(0);
 [train_x,val_x,test_x,train_y,val_y,test_y] = setupmnist(0.1);
 [train_x_all,~,~,~,~,~] =setupmnist(1);   % load full dataset
 %create semisup x
 start_idx = size(train_x) + 1;
 semisup_x = train_x_all(start_idx:end,:);
 disp(['Number of semisup samples: ', num2str(size(semisup_x,1))])
 
 
 
 f = fullfile(pwd,[name '.mat'])

% Setup DBN
sizes = [3000 ];   % hidden layer size

opts = dbncreateopts();
opts.x_semisup = semisup_x;
opts.alpha = 0.01; % 0 = discriminative, 1 = generative
opts.beta = 0.1;
opts.gpu   = 1;                  % use GPU other optsion are 0: CPU, -1: CPU test
opts.cdn = 1;   
opts.sparsity = 10^-4;
opts.thisgpu = gpu;              % ref to gpu,  must be set if opts.gpu =1
opts.gpubatch = size(train_x,1); 
opts.outfile = [name '_intermediate.mat'];
opts.patience = 15;
opts.numepochs = 1000;
opts.testinterval = 1;
opts.init_type = 'cRBM';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = val_x;
opts.y_val = val_y;
opts.traintestbatch = 10000;

%% Set learningrate and momentum
opts.learningrate = @(t,momentum) 0.05;
opts.momentum = @(t) 0;


[dbn, opts]  = dbnsetup(sizes, train_x, opts);  % train function 

rbm = dbn.rbm{1};
opts.gpu
opts.numepochs
disp(rbm);

fprintf('\n\n')

rbm = rbmtraingpu(rbm,train_x,opts);

 save(f,'rbm','opts');
% 
% 
% class_vec = zeros(100,size(train_y,2));
% for i = 1:size(train_y,2)
%     class_vec((i-1)*10+1:i*10,i) = 1;
% end
% digits = dbnsample(dbn,100,10000,class_vec);
% 
% 
% 
% 

