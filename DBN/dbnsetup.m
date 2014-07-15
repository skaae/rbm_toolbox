function dbn = dbnsetup(dbn, x, opts)
%%DBNSETUP creates a propr dbn struct
% INPUT: dbn : a struct with field sizes which should be a vector of RBM
%              sizes. sizes(1) is the size fo the hidden layer of the first RBM
%              sizes(2) is hidden layer of 2. rbm etc
%        x   : used to specify size of first hidden layer
%        opts: a struct with options see below
%
% modified june 2014 by Søren Sønderby
n = size(x, 2);
dbn.sizes = [n, dbn.sizes];
n_rbm = numel(dbn.sizes) - 1;


valid = @(f) isfield(opts,'x_val') == 1 && ~isempty(opts.(f));
reqboth =@(a,b) valid(a) && ~valid(b);

% require that they are on togheter
if  reqboth('y_val','x_val')
    error('For validation specify both y_val and x_val')
end


% check if y is given if class rbm
if valid('classRBM') && opts.classRBM == 1
    if ~valid('y_train')
        error('classRBM  requires y_train to be specified in opts')
    elseif reqboth('y_val','x_val')
        error('classRBM with x_val must also specify y_val in opts')
    end
end

% if early stopping a validation set must be specified
if opts.early_stopping
    if opts.classRBM == 1
        if  reqboth('y_val','x_val')
            error('Eearly stopping with classRBM requires x and y val sets')
        end
    end
elseif opts.classRBM == 0
    error('Early stopping is not implemented for generativeRBM')
    
end

% if discriminitive training require labels
if isequal(opts.train_func,@rbmdiscriminative)
    if ~valid('y_train')
        error('Discriminative training requires labels')
    end
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
    dbn.rbm{u}.L2 = opts.L2;
    dbn.rbm{u}.L1 = opts.L1;
    dbn.rbm{u}.L2norm = opts.L2norm;
    dbn.rbm{u}.sparsity = opts.sparsity;
    dbn.rbm{u}.error = [];
    dbn.rbm{u}.val_error = [];
    dbn.rbm{u}.train_error  = [];
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
        n_classes = size(opts.y_train,2);
        dbn.rbm{u}.U  = normrnd(0,0.01,hid_size, n_classes);
        dbn.rbm{u}.vU  = normrnd(0,0.01,hid_size, n_classes);
        
        dbn.rbm{u}.d  = normrnd(0,0.01,n_classes, 1);
        dbn.rbm{u}.vd  = normrnd(0,0.01,n_classes, 1);
        
    else
        dbn.rbm{u}.classRBM = 0;
        dbn.rbm{u}.U  = [];
        dbn.rbm{u}.vU  = [];
        dbn.rbm{u}.d  = [];
        dbn.rbm{u}.vd  = [];
    end
    
    
    
    dbn.rbm{u}.W  = normrnd(0,0.01,hid_size, vis_size);
    dbn.rbm{u}.vW = zeros(hid_size, vis_size);
    
    dbn.rbm{u}.b  = normrnd(0,0.01,vis_size, 1);
    dbn.rbm{u}.vb = zeros(vis_size, 1);
    
    dbn.rbm{u}.c  = normrnd(0,0.01,hid_size, 1);
    dbn.rbm{u}.vc = zeros(hid_size, 1);
end

end
