function dbn = dbntrain(dbn, x_train, opts)
%%DBNTRAIN train a DBN by stacking RBM's
% see dbncreateopts for a descriptionof the opts struct. Use dbnsetup
% to create the dbn struct
%
% modified june 2014 by Søren Sønderby



    
    


n_rbm = numel(dbn.rbm);

line = repmat('-',1,80);
fprintf('%s\n                  TRAINING RBM 1\n %s\n',line,line);
dbn.rbm{1} = rbmtrain(dbn.rbm{1},x_train,opts);

for i = 2 : n_rbm
    
    fprintf('%s\n                  TRAINING RBM %i\n %s\n',line,i,line);
    x_train = rbmup(dbn.rbm{i - 1}, x_train,@sigm);
    
    if ~isempty(opts, 'x_val')
        opts.x_val = rbmup(dbn.rbm{i - 1}, opts.x_val,@sigm);
    end
        
    if i == n_rbm && opts.hintonDBN == 1
        % for last DBN we need to assign the  class labels to the 
        % visible states of the last RBM
        x_train = [x_train opts.y_train];
        if ~isempty(opts, 'x_val')
            opts.x_val = [opts.x_val opts.y_val];
        end
        
    
    end
    
    dbn.rbm{i} = rbmtrain(dbn.rbm{i},x_train,opts);
    
    
    
end



end
