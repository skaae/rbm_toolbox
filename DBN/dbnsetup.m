function [dbn, opts] = dbnsetup(sizes, x, opts)
%%DBNSETUP creates a propr dbn struct
%     INPUT
%        sizes : A vector with hidden layer sizes
%            x : used to specify size of first hidden layer
%         opts : a struct with options see dbncreateopts
%
% See also DBNCREATEOPTS DBNTRAIN DBNCHECKOPTS
%
% Copyright Søren Sønderby july 2014

[n_samples, n] = size(x);
dbn.sizes = [n, sizes];
n_rbm = numel(dbn.sizes) - 1;



% create weight initialization function
if isa(opts.inittype, 'function_handle')
    initfunc = opts.inittype;
elseif ischar(opts.inittype)
    switch lower(opts.inittype)
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


% if traintestbatch is not given set it to the complete training set
if isempty(opts.traintestbatch )
    opts.traintestbatch = size(x,1);
end


%% use lr function etc to calculate learning rates

% convert scalars to functions
opts.cdn = create_func(opts.cdn,1);
opts.learningrate = create_func(opts.learningrate,2);
opts.momentum = create_func(opts.momentum,1);

% calculate values
mom = arrayfun(opts.momentum,1:opts.numepochs);
lr = arrayfun(opts.learningrate,1:opts.numepochs,mom);
cdn = arrayfun(opts.cdn,1:opts.numepochs);


for u = 1 : n_rbm
    if u == n_rbm
        % init bias and weights for class vectors
        cur_n_classes = size(opts.y_train,2);
        cur_alpha = opts.alpha;
    else
        % for non toplayers use generative training
        cur_n_classes = 1;
        cur_alpha = 1;  %generative training for non toplayers
    end
    
    dbn.rbm{u}.testinterval = opts.testinterval;
    dbn.rbm{u}.traintestbatch =   opts.traintestbatch;
    
    dbn.rbm{u}.cdn = cdn;
    dbn.rbm{u}.learningrate = lr;
    dbn.rbm{u}.momentum = mom;
    dbn.rbm{u}.alpha = cur_alpha;
    dbn.rbm{u}.beta = opts.beta;
    
    % regularization parameters
    dbn.rbm{u}.L2 = opts.L2;
    dbn.rbm{u}.L1 = opts.L1;
    dbn.rbm{u}.sparsity = opts.sparsity;
    dbn.rbm{u}.dropout = opts.dropout;
    dbn.rbm{u}.dropconnect = opts.dropconnect;
    
    % error stuff
    dbn.rbm{u}.errfunc = opts.errfunc;
    dbn.rbm{u}.reconerror = [];
    dbn.rbm{u}.valerror = [];
    dbn.rbm{u}.trainerror  = [];
    dbn.rbm{u}.trainerrormeasures = {};
    dbn.rbm{u}.valerrormeasures = {};
    dbn.rbm{u}.patience = opts.patience;
    
    
    vis_size =  dbn.sizes(u);
    hid_size = dbn.sizes(u + 1);
    
    dbn.rbm{u}.U  = initfunc(hid_size, cur_n_classes);
    dbn.rbm{u}.vU  = zeros(hid_size, cur_n_classes);
    
    dbn.rbm{u}.d  = zeros(cur_n_classes, 1);
    dbn.rbm{u}.vd  = zeros(cur_n_classes, 1);

    
    
    dbn.rbm{u}.W  = initfunc(hid_size, vis_size);
    dbn.rbm{u}.vW = zeros(hid_size, vis_size);
    dbn.rbm{u}.b  = zeros(vis_size,1);
    dbn.rbm{u}.vb = zeros(vis_size, 1);
    dbn.rbm{u}.c = zeros(hid_size,1);
    dbn.rbm{u}.vc = zeros(hid_size, 1);
    
    
    
    
    %% functions depending on cpu / gpu / testing
    switch opts.gpu
        case 0
            dbn.rbm{u}.gpubatch= n_samples;
            dbn.rbm{u}.rand    = @rand;
            dbn.rbm{u}.randi    = @randi;
            dbn.rbm{u}.zeros   = @zeros;
            dbn.rbm{u}.ones    = @ones;
            dbn.rbm{u}.cpToGPU = @(hrbm) hrbm;
            dbn.rbm{u}.cpToHOST= @(drbm) drbh;
            dbn.rbm{u}.cpWeightsToHOST= @(drbm) @cpWeightstoHOST;
            dbn.rbm{u}.array   = @(x) double(x);
            dbn.rbm{u}.gather   = @(x) x;
            dbn.rbm{u}.colon   = @colon;
            
        case 1
            dbn.rbm{u}.gpubatch= opts.gpubatch;
            dbn.rbm{u}.rand    = @gpuArray.rand;
            dbn.rbm{u}.randi    = @gpuArray.randi;
            dbn.rbm{u}.zeros   = @gpuArray.zeros;
            dbn.rbm{u}.ones    = @gpuArray.ones;
            dbn.rbm{u}.cpToGPU = @cpRBMtoGPU;
            dbn.rbm{u}.cpToHOST= @cpRBMtoHost;
            dbn.rbm{u}.cpWeightsToHOST= @(drbm) @cpWeightstoHOST;
            dbn.rbm{u}.array   = @(x) gpuArray(x);
            dbn.rbm{u}.gather   = @(x) gather(x);
            dbn.rbm{u}.colon   = @gpuArray.colon;
        case -1   % for testing
            dbn.rbm{u}.gpubatch=opts.gpubatch;
            dbn.rbm{u}.rand    = @(val) test1(val,@rand);
            dbn.rbm{u}.randi   = @(val1,val2) test2(val1,val2,@randi);
            dbn.rbm{u}.zeros   = @zeros;
            dbn.rbm{u}.ones    = @ones;
            dbn.rbm{u}.cpToGPU = @(hrbm) hrbm;
            dbn.rbm{u}.cpToHOST= @(drbm) drbh;
            dbn.rbm{u}.cpWeightsToHOST= @(drbm) @cpWeightstoHOST;
            dbn.rbm{u}.array   = @(x) double(x);
            dbn.rbm{u}.gather   = @(x) x;
            dbn.rbm{u}.colon   = @colon;
            
            
        otherwise
            ('Unkwnown opts.gpu setting')
    end
    
    
    init = dbn.rbm{u}.array;
    switch upper(opts.traintype)
        case 'CD'
             dbn.rbm{u}.traintype = 0;
             dbn.rbm{u}.pcdchainsx = 0;
             dbn.rbm{u}.pcdchainsy = 0;
             if opts.beta > 0
                %init semisup chains
                dbn.rbm{u}.pcdchainsx_semisup =0;
                dbn.rbm{u}.pcdchainsy_semisup = 0;
            end
        case 'PCD'
            % init chains
            dbn.rbm{u}.traintype = 1;
            kk = randperm(n_samples);
            kk = kk(1:opts.npcdchains);
            dbn.rbm{u}.pcdchainsx = x(kk,:);
            dbn.rbm{u}.pcdchainsy = opts.y_train(kk,:);
            if opts.beta > 0
                %init semisup chains
                kk_semisup = randperm(size(opts.x_semisup,1));
                kk_semisup = kk_semisup(1:opts.npcdchains);
                dbn.rbm{u}.pcdchainsx_semisup = opts.x_semisup(kk_semisup,:);
                dbn.rbm{u}.pcdchainsy_semisup = opts.y_train(kk,:);
            end
        otherwise
            error('Traintype must be CD or PCD')
    end
    
    
    
    
    
end

    function f = test1(val,func)
        rng('default');rng(101);
        f = func(val);
    end

    function f = test2(val1,val2,func)
        rng('default');rng(101);
        f = func(val1,val2);
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

    function ret = create_func(val,nargs)
        % takes a scalar val or function handle and returns a function returning
        % val if val is not a function.
        if isa(val, 'function_handle')
            ret = val;
        else  % assume its a scalar
            switch nargs
                case 1
                    ret = @(v1) val;
                case 2
                    ret = @(v1,v2) val;
            end
        end
    end

end
