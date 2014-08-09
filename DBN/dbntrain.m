function dbn = dbntrain(dbn, x_train, opts)
%%DBNTRAIN train a DBN by stacking RBM's
% see dbncreateopts for a description of the opts struct. Use dbnsetup
% to create the dbn struct.
%
% Copyright june 2014 by Søren Sønderby
rows = @(x) size(x,1);
cols = @(x) size(x,2);

n_rbm = numel(dbn.rbm);
n_classes = cols(dbn.rbm{1}.U);
n_xt      = rows(x_train);
n_xv      = ifelse(~isempty(opts.x_val),rows(opts.x_val),[]);
n_xs      = ifelse(~isempty(opts.x_semisup),rows(opts.x_semisup),[]);
line = repmat('-',1,80);
fprintf('%s\n                  TRAINING RBM 1\n %s\n',line,line);

if n_rbm == 1
    dbn.rbm{1} = rbmtraingpu(dbn.rbm{1},x_train,opts);
else
    
    % copy  original y values
    y_org_train = opts.y_train;
    if ~isempty(opts.x_val)
        y_org_val = opts.y_val;
    end
    
    %% set y values to zeros for non top rbm
    opts.y_train = zeros( n_xt,1 );
    if ~isempty(opts.x_val)
        opts.y_val = zeros( n_xv,1 );
    end
    
    %% train bottom rbm
    dbn.rbm{1} = rbmtraingpu(dbn.rbm{1},x_train,opts);
    
    
    
    % iterate over the rest of the rbms
    for i = 2 : n_rbm
        prev_rbm = dbn.rbm{i - 1};
        curr_rbm = dbn.rbm{i};
        
        fprintf('%s\n                  TRAINING RBM %i\n %s\n',line,i,line);
        % pass up x_train
        x_train = rbmup(prev_rbm,x_train,zeros(n_xt,n_classes),@sigm);
        
        % pass up x_val
        if ~isempty(opts.x_val)
            opts.x_val = rbmup(prev_rbm, opts.x_val,zeros(n_xv,n_classes),@sigm);
        end
        
        % pass up x_semisup
        if prev_rbm.beta > 0
            opts.x_semisup = rbmup(prev_rbm, opts.x_semisup,zeros(n_xs,n_classes),@sigm);
        end
        
        
        
        %% check if it is the top rbm, if so use original y values
        % then train the current rbm
        if  n_rbm == i
            opts.y_train = y_org_train;
            if ~isempty(opts.x_val)
                opts.y_val = y_org_val;
            end
        else   % not top rbm
            opts.y_train = zeros( n_xt,1 );
            if ~isempty(opts.x_val)
                opts.y_val = zeros( n_xv,1 );
            end
        end
        
        dbn.rbm{i} = rbmtraingpu(curr_rbm,x_train,opts);
    end
    
end










end
