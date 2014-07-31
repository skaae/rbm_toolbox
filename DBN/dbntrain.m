function dbn = dbntrain(dbn, x_train, opts)
%%DBNTRAIN train a DBN by stacking RBM's
% see dbncreateopts for a description of the opts struct. Use dbnsetup
% to create the dbn struct.
%
% EXAMPLE
%  sizes = [200];   % hidden layer size
%  [opts, valid_fields] = dbncreateopts();
%  dbncheckopts(opts,valid_fields);       
%  disp(opts)
%  dbn = dbnsetup(sizes, train_x, opts); 
%  dbn = dbntrain(dbn, train_x, opts);
%
% See also DBNCREATEOPTS DBNCHECKOPTS DBNSETUP DBNTRAIN
%
% modified june 2014 by Søren Sønderby
n_rbm = numel(dbn.rbm);

line = repmat('-',1,80);
fprintf('%s\n                  TRAINING RBM 1\n %s\n',line,line);

dbn.rbm{1} = rbmtrain(dbn.rbm{1},x_train,opts);


for i = 2 : n_rbm
    
    if dbn.rbm{i}.classRBM == 1 && n_rbm == i
        ye = opts.y_train;
    else
        ye = [];
    end
    
    
    fprintf('%s\n                  TRAINING RBM %i\n %s\n',line,i,line);
    x_train = rbmup(dbn.rbm{i - 1},x_train,ye,@sigm);
    
    if ~isempty(opts.x_val)
        opts.x_val = rbmup(dbn.rbm{i - 1}, opts.x_val,ye,@sigm);
    end
    
    dbn.rbm{i} = rbmtrain(dbn.rbm{i},x_train,opts);
    
    
    
end



end
