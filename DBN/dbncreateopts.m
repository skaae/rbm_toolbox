function opts  = dbncreateopts()
%DBNCREATEOPTS creates a valid opts struct
% The OPTS struct
%    The following fields are valid
%
%               cdn : integer. Number of gibbs steps.
%                     Applies to both CD and PCD setting. see [3,4]
%         numepochs : number of epochs
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
%          sparsity : Use a simple sparsity measure. substract sparsity from the
%                     hidden biases after each update. see [1]
%         classRBM  : If this field exists and is 1 then train the DBN where the
%                     visible layer of the last RBM has the training labels
%                     added. See "To recognize shapes, first learn to generate
%                     images" Requires y_train to be spcified.
%      testinterval : how often the performance should be measured
%           y_train : Must be specified if classRBM is 1
%             x_val : If specified the energy ratio between a training set the
%                     and the validation set will be caluclated every
%                     ratio_interval epoch
%             y_val : if classRBM is a field and x_val is a field this field
%                     must be specified
%         x_semisup : unsupervised training examples. For use when the training
%                     function is @rbmsemisuplearn
%          patience : Patience when using early stopping. Notice that 
%                     epochs that will pass before we stop are 
%                     patience * test_interval. E.g i you want a patience of 
%                     5 epocs and the test_interval is 5 set patience to 1
%                     
%          errfunc : A function which return a error measure. This applies only
%                     to a classRBM. The error function 
%                     takes the predicted probabilites as first argument and the
%                     one-of-K encoded true labels as second arguments. see
%                     accuracy.m in utils folder.
%                   
%            alpha : weigthing of generative and hybrid training objective see 
%                    [1]
%             beta : importance of unupservised samples in semi-supervised
%                    learning.
%    dropouthidden : dropout fraction of hidden units.
%         inittype : initialization of weightes. 
%                     'gauss' init at gaussian with 0 mean and 0.01 std 
%                     'cRBM' init as larochelle in [1] i.e  
%                     weights = randnd(size(weights))-0.5 ./ max(size(weights)).
%                     Bias units are always initialized at zero.
%           outfile : after each epoch the best_rbm or rbm is saved to this file
%               gpu : 1 for gpu, 0 for cpu, -1 for testing on cpu
%          gpubatch : load batches of this size to gpu. If you have mem probles
%                     try with a lower value
%            thisgpu: reference to current gpu if opts.gpu is 1. 
%          traintype: Contrastive divergence (CD) or Persistent CD (PCD)
%         npcdchains: number of pcd chains
% References
%     [1] H. Larochelle and M. Mandel, ?Learning algorithms for the
%         classification restricted boltzmann machine,? J. Mach.  ?, 2012.      
%     [2] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and 
%         R. R. Salakhutdinov, ?Improving neural networks by preventing 
%         co-adaptation of feature detectors,? Jul. 2012
%     [3] T. Tieleman, ?Training restricted Boltzmann machines using 
%         approximations to the likelihood gradient,? ? 
%         25th Int. Conf. Mach. ?, 2008.
%     [4] G. Hinton, ?Training products of experts by minimizing contrastive 
%         divergence,? Neural Comput., 2002.
%     [5] N. Srivastava and G. Hinton, ?Dropout: A Simple Way to Prevent Neural
%         Networks from Overfitting,? J. Mach.  ?, 2014.
%
% see also, DBNSETUP, DBNCHECKOPTS DBNTRAIN
%
% Copyright Søren Sønderby july 2014


% DEFAULT SETTINGS
opts.traintype = 'CD';
opts.npcdchains = 100;
opts.testinterval = 5;
opts.numepochs =   100;
opts.cdn = 1;

T = 50;       % momentum ramp up
p_f = 0.9;    % final momentum
p_i = 0.5;    % initial momentum
eps = 0.01;    % initial learning rate
f = 0.9;     % learning rate decay
opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum);
opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);
%opts.momentum     = @(t) 0;
opts.gpubatch = 10000;
opts.L1 = 0;
opts.L2 = 0;
opts.sparsity = 0;
opts.classRBM = 1;
opts.y_train = [];
opts.x_val = [];
opts.y_val = [];
opts.x_semisup = [];
opts.patience = 5;
opts.alpha = 0;
opts.beta = 0;
opts.errfunc = @accuracy;
opts.dropouthidden = 0;
opts.inittype = 'gauss';
opts.outfile = [];
opts.gpu = 0;

end

