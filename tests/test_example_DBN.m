if ~ismac
    cd('/zhome/f9/4/69552/DeepLearnToolbox_noGPU')
    addpath(genpath('/zhome/f9/4/69552/DeepLearnToolbox_noGPU'))
    addpath('~/dataanalysis-master/')

end


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


%%  Generative RBM with 200 hidden units and contrastive divergence
% note that the learningrate and the momentum must be specified as functions.
rng('default');rng(0);
load mnist_uint8;
train_x = double(train_x)/255;
train_y = double(train_y);


sizes = [200];   % hidden layer size
[opts, valid_fields] = dbncreateopts();

opts.numepochs = 30;
opts.traintype = 'CD';
opts.classRBM = 0;
opts.y_train = train_y;
opts.test_interval = 1;
opts.train_func = @rbmgenerative;
opts.init_type = 'cRBM';

opts.learningrate = @(t,momentum) 0.05;
opts.momentum     = @(t) 0;

dbncheckopts(opts,valid_fields);       
disp(opts)
dbn = dbnsetup(sizes, train_x, opts); 
dbn = dbntrain(dbn, train_x, opts);

% visualize weights
figure;visualize(dbn.rbm{1}.W(1:144,:)'); 
set(gca,'visible','off');

%% example 2 classification generative RBM with CD training
clear all
rng('default');rng(0);
load mnist_uint8;
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y = double(test_y);

sizes = [200];  
[opts, valid_fields] = dbncreateopts();
opts.early_stopping = 1;
opts.patience = 5;

opts.numepochs = 50;
opts.traintype = 'CD';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = test_x;
opts.y_val = test_y;
opts.test_interval = 1;
opts.train_func = @rbmgenerative;
opts.init_type = 'cRBM';

opts.learningrate = @(t,momentum) 0.05;
opts.momentum     = @(t) 0;

dbncheckopts(opts,valid_fields);       
disp(opts)
dbn = dbnsetup(sizes, train_x, opts);  
dbn = dbntrain(dbn, train_x, opts);

% Make predictions
pred_val = dbnpredict(dbn,test_x);
[~, labels_val] = max(test_y,[],2);
acc_val = mean(pred_val == labels_val);
err_val = 1-acc_val

% plot weights 
figure;visualize(dbn.rbm{1}.W(1:144,:)'); 
set(gca,'visible','off');


% plot errors
plot([dbn.rbm{1}.val_error',dbn.rbm{1}.train_error'])
legend({'Validation error','Train error'})
[min_val,min_idx] =  min(dbn.rbm{1}.val_error);
hold on; plot(min_idx,min_val,'xr'); hold off;
xlabel('Epoch'); ylabel('Error'); grid on;

%% Example 3 - PCD and sampling and movies
clear all;
rng('default');rng(0);
load mnist_uint8;
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y = double(test_y);

sizes = [200 200];   % hidden layer size
[opts, valid_fields] = dbncreateopts();
opts.numepochs = 100;
opts.traintype = 'PCD';
opts.classRBM = 1;
opts.y_train = train_y;
opts.x_val = test_x;
opts.y_val = test_y;
opts.test_interval = 1;
opts.train_func = @rbmgenerative;

%% Set learningrate
eps       		  = 0.001;    % initial learning rate
f                 = 0.99;      % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);

% Set momentum
T             = 50;       % momentum ramp up
p_f 		  = 0.9;    % final momentum
p_i           = 0.5;    % initial momentum
opts.momentum = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);


dbncheckopts(opts,valid_fields);       %checks for validity of opts struct
dbn = dbnsetup(sizes, train_x, opts);  % train function 
dbn = dbntrain(dbn, train_x, opts);

class_vec = zeros(100,size(train_y,2));
for i = 1:size(train_y,2)
    class_vec((i-1)*10+1:i*10,i) = 1;
end

digits = dbnsample(dbn,100,10000,class_vec); 


