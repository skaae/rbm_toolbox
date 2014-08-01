function [ opts,valid_fields ] = dbncreateopts()
%DBNCREATEOPTS creates a valid opts struct
% The OPTS struct
%    The following fields are valid
%
%         traintype : CD for contrastive divergence, PCD for persistent
%                     contrastive divergence. see [3,4]
%               cdn : integer. Number of gibbs steps.
%                     Applies to both CD and PCD setting. see [3,4]
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
%                     weights for this neuron is rescaled to L2norm. See [2]
%          sparsity ; Use a simple sparsity measure. substract sparsity from the
%                     hidden biases after each update. see [1]
%         classRBM  : If this field exists and is 1 then train the DBN where the
%                     visible layer of the last RBM has the training labels
%                     added. See "To recognize shapes, first learn to generate
%                     images" Requires y_train to be spcified.
%    test_interval  : how often the performance should be measured
%           y_train : Must be specified if classRBM is 1
%             x_val : If specified the energy ratio between a training set the
%                     and the validation set will be caluclated every
%                     ratio_interval epoch
%             y_val : if classRBM is a field and x_val is a field this field
%                     must be specified
%         x_semisup : unsupervised training examples. For use when the training
%                     function is @rbmsemisuplearn
%    early_stopping : Use earlystopping
%          patience : Patience when using early stopping. Notice that 
%                     epochs that will pass before we stop are 
%                     patience * test_interval. E.g i you want a patience of 
%                     5 epocs and the test_interval is 5 set patience to 1
%        train_func : @rbmgenerative: Generative rbm training with or without 
%                     labels. 
%                     @rbmdiscriminative: discriminative training. Requires
%                     training labels. 
%                     @rbmhybrid mix of generative and discriminative, see [1]
%                     @rbmsemisublearn use unsupervised training. See [1] sec 8
%                     requires x_unsup to be set. control importance of 
%                     unsupervised training with the beta param
%                     The semi_sup_type param determines if semisupervised 
%                     training is combined with hybrid, generative or 
%                     discriminative training.
%                     
%          err_func : A function which return a error measure. This applies only
%                     to a classRBM. The error function 
%                     takes the predicted probabilites as first argument and the
%                     one-of-K encoded true labels as second arguments. see
%                     accuracy.m in utils folder.
%                   
%      hybrid_alpha : weigthing of generative and hybrid training objective see 
%                     [1]
%      semisup_beta : importance of unupservised samples in semi-supervised
%                     learning.
%      semisup_type : either @rbmhybrid, @rbmgenerative or @rbmdiscriminative
%                     see train_func for description.
%    dropout_hidden : dropout fraction of hidden units.
%         init_type : initialization of weightes. 
%                     'gauss' init at gaussian with 0 mean and 0.01 std 
%                     'cRBM' init as larochelle in [1] i.e  
%                     weights = randnd(size(weights))-0.5 ./ max(size(weights)).
%                     Bias units are always initialized at zero.
%              gpu  : 0|1 use gpu for calculation
%
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
opts.traintype = 'PCD';
opts.numepochs =   100;
opts.batchsize = 100;
opts.cdn = 1;

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
opts.sparsity = 0;
opts.classRBM = 0;
opts.test_interval = 5;
opts.y_train = [];
opts.x_val = [];
opts.y_val = [];
opts.x_semisup = [];
opts.early_stopping = 0;
opts.patience = 5;
opts.train_func = @rbmgenerative;
opts.hybrid_alpha = 0.5;
opts.semisup_beta = 0.1;
opts.semisup_type = @rbmhybrid;
opts.err_func = @accuracy;
opts.dropout_hidden = 0;
opts.init_type = 'gauss';
opts.gpu = 0;

valid_fields = fieldnames(opts);

end

