function dbn = dbnsetup(sizes, x, opts)
%%DBNSETUP creates a propr dbn struct
%     INPUT
%        sizes : A vector with hidden layer sizes
%            x : used to specify size of first hidden layer
%         opts : a struct with options see dbncreateopts
%
% See also DBNCREATEOPTS DBNTRAIN DBNCHECKOPTS
%
% Copyright Søren Sønderby july 2014

n = size(x, 2);
dbn.sizes = [n, sizes];
n_rbm = numel(dbn.sizes) - 1;



% create weight initialization function
if isa(opts.init_type, 'function_handle')
    initfunc = opts.init_type;
elseif ischar(opts.init_type)
    switch lower(opts.init_type)
        case 'gauss'
            initfunc = @(m,n) normrnd(0,0.01,m,n);
        case 'crbm'
            initfunc = @init_crbm;
        otherwise
            error('init_type should be either gauss or cRBM');
    end
else
    error('opts.init_type must be function handle or the strings gaus/cRBM');
end

for u = 1 : n_rbm
    
    
    
    % if one learningrate/momentum function use this for all
    % otherwise use individual learningrate/momentum for each rbm
    if numel(opts.learningrate) == n_rbm && n_rbm ~=1
        dbn.rbm{u}.learningrate = opts.learningrate{u};
    else
        assert(numel(opts.learningrate)==1,'learnfunc. should be 1 or nrbm')
        dbn.rbm{u}.learningrate = opts.learningrate;
    end
    
    if numel(opts.momentum) == n_rbm && n_rbm ~= 1
        dbn.rbm{u}.momentum = opts.momentum{u};
    else
        assert(numel(opts.momentum)==1,'Momen. func should be 1 or nrbm')
        dbn.rbm{u}.momentum = opts.momentum;
    end
    
    % regularization parameters
    dbn.rbm{u}.L2 = opts.L2;
    dbn.rbm{u}.L1 = opts.L1;
    dbn.rbm{u}.L2norm = opts.L2norm;
    dbn.rbm{u}.sparsity = opts.sparsity;
    dbn.rbm{u}.dropout_hidden = opts.dropout_hidden;
    
    % error stuff
    dbn.rbm{u}.err_func = opts.err_func;
    dbn.rbm{u}.error = [];
    dbn.rbm{u}.val_error = [];
    dbn.rbm{u}.train_error  = [];
    dbn.rbm{u}.train_error_measures = {};
    dbn.rbm{u}.val_error_measures = {};
    dbn.rbm{u}.energy_ratio = [];
    
    
    
    % i havent implemented early stopping for non top layers because
    % they are not classRBMS
    if n_rbm == u
        dbn.rbm{u}.early_stopping = opts.early_stopping;
    else
        dbn.rbm{u}.early_stopping = 0;
    end
    dbn.rbm{u}.patience = opts.patience;
    
    
    
    
    vis_size =  dbn.sizes(u);
    hid_size = dbn.sizes(u + 1);
    
    if opts.classRBM == 1 && u == n_rbm
        % init bias and weights for class vectors
        dbn.rbm{u}.classRBM = 1;
        
        dbn.rbm{u}.train_func = opts.train_func;
        n_classes = size(opts.y_train,2);
        dbn.rbm{u}.U  = initfunc(hid_size, n_classes);
        dbn.rbm{u}.vU  = zeros(hid_size, n_classes);
        
        dbn.rbm{u}.d  = zeros(n_classes, 1);
        dbn.rbm{u}.vd  = zeros(n_classes, 1);
        
    else
        % for non toplayers use generative training
        dbn.rbm{u}.classRBM = 0;
        dbn.rbm{u}.train_func = @rbmgenerative;
        
        dbn.rbm{u}.U  = [];
        dbn.rbm{u}.vU  = [];
        dbn.rbm{u}.d  = [];
        dbn.rbm{u}.vd  = [];
        
        
    end
    
    
    
    dbn.rbm{u}.W  = initfunc(hid_size, vis_size);
    dbn.rbm{u}.vW = zeros(hid_size, vis_size);
    
    
    %dbn.rbm{u}.b  = normrnd(0,0.01,vis_size, 1);
    dbn.rbm{u}.b  = zeros(vis_size,1);
    dbn.rbm{u}.vb = zeros(vis_size, 1);
    
    %dbn.rbm{u}.c  = normrnd(0,0.01,hid_size, 1);
    dbn.rbm{u}.c = zeros(hid_size,1);
    dbn.rbm{u}.vc = zeros(hid_size, 1);
    
    
    %%% prepare for GPU
    %set gpu info
    dbn.rbm{u}.gpu = opts.gpu;
    
    % for non class RBM's rbmy should return empty. To avoid if statement
    % create a functio returning empty otherwise use rbmdowny
    
    if dbn.rbm{u}.gpu
        dbn.rbm{u}.rand =  @gpuArray.rand;
        dbn.rbm{u}.zeros = @gpuArray.zeros;
    else
        dbn.rbm{u}.rand    = @rand;
        dbn.rbm{u}.zeros    = @zeros;
        
    end
    
    
    if dbn.rbm{u}.classRBM
        dbn.rbm{u}.rbmdowny = @rbmdownyclassrbm;
        dbn.rbm{u}.rbmup    = @rbmupclassrbm;
    else
        dbn.rbm{u}.rbmdowny = @rbmdownynotclass;
        dbn.rbm{u}.rbmup    = @rbmupnotclassrbm;
        
    end
    
end

    function weights = init_crbm(m,n)
        % initilize weigts from uniform distribution. As described in
        % Learning Algorithms for the Classification Restricted Boltzmann
        % machine
        M = max([m,n]);
        interval_max = M^(-0.5);
        interval_min = -interval_max;
        weights = interval_min + (interval_max-interval_min).*rand(m,n);
        
        assert(max(weights(:)) <= interval_max)
        assert(min(weights(:)) >= interval_min)
    end

end
