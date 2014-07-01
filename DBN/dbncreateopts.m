function [ opts ] = dbncreateopts()
%DBNCREATEOPTS creates a valid opts struct
% The OPTS struct
%    The following fields are valid
%
%         traintype : 'CD' for contrastive divergence, 'PCD' for persistent
%                      contrastive divergence.
%               cdn : integer. Number of gibbs steps before negative statistics 
%                     is collected. Applies to both CD and PCD setting
%         numepochs : number of epochs
%         batchsize : minibatch size. mod(n_samples,batchsize) must be 0
%      learningrate : a function taking current epoch and current momentum as 
%                     as variables and returns a learning rate e.g.
%                     @(epoch,momentum) 0.1*0.9.^epoch*(1-momentum)
%          momentum : a function that takes epoch number as input and returns a 
%                     momentum rate e.g.
%                     T = 50;       % momentum ramp up
%                     p_f = 0.9;    % final momentum
%                     p_i = 0.5;    % initial momentum
%                     @(epoch)ifelse(epoch<T,p_i*(1-epoch/T)+(epoch/T)*p_f,p_f)
%                L1 : double specifying L1 weight decay 
%                L2 : double specifying L2 weight decay
%            L2norm : double specifying constraint on the incoming weight sizes
%                     to each nuron. If the L2norm is above this value the
%                     weights for this neuron is rescaled to L2norm. See 
%                     http://arxiv.org/abs/1207.0580
%         hintonDBN : If this field exists and is 1 then train the DBN where the 
%                     visible layer of the last RBM has the training labels 
%                     added. See "To recognize shapes, first learn to generate 
%                     images" Requires y_train to be spcified.
%           y_train : Must be specified if hintonDBN is a field
%             x_val : If specified the energy ratio between a training set the
%                     and the validation set will be caluclated every 5th epoch
%             y_val : if hintonDBN is a field and x_val is a field this field
%                     must be specified
%

dbn.sizes = [500 500 2000];
opts.traintype = 'PCD';
opts.numepochs =   100;
opts.batchsize = 100;
opts.cdn = 1; % contrastive divergence

T = 50;       % momentum ramp up
p_f = 0.9;    % final momentum
p_i = 0.5;    % initial momentum
eps = 0.01;    % initial learning rate
f = 0.9;     % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum); 
opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);
%opts.momentum     = @(t) 0;
opts.L1 = 0.00;
opts.L2 = 0;
opts.L2norm = 0;

opts.hintonDBN = 0;
opts.y_train = [];
opts.x_val = [];
opts.y_val = [];
end

