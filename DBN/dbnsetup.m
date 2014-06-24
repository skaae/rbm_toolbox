function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];
    n_rbm = numel(dbn.sizes) - 1;
    
    
    
    for u = 1 : n_rbm
        
        % if one learningrate/momentum function use this for all 
        % otherwise use individual learningrate/momentum for each rbm
        if numel(opts.learningrate) == n_rbm
            dbn.rbm{u}.learningrate = opts.learningrate{u};
        else
            assert(numel(opts.learningrate)==1,'learnfunc. should be 1 or nrbm')
            dbn.rbm{u}.learningrate = opts.learningrate;
        end
        
        if numel(opts.momentum) == n_rbm
            dbn.rbm{u}.momentum = opts.momentum{u};
        else
            assert(numel(opts.momentum)==1,'Momen. func should be 1 or nrbm')
            dbn.rbm{u}.momentum = opts.momentum;
        end
        
        dbn.rbm{u}.W  = normrnd(0,0.01,dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = normrnd(0,0.01,dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = normrnd(0,0.01,dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
