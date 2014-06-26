function dbn = dbntrain(dbn, x_train, opts,x_test)
n = numel(dbn.rbm);
if nargin == 3
    dbn.rbm{1} = rbmtrain(dbn.rbm{1},x_train,opts);
elseif nargin == 4
    dbn.rbm{1} = rbmtrain(dbn.rbm{1},x_train,opts,x_test);
end

for i = 2 : n
    x_train = rbmup(dbn.rbm{i - 1}, x_train,@sigm);
    
    if nargin == 3
        dbn.rbm{i} = rbmtrain(dbn.rbm{i},x_train,opts);
    elseif nargin == 4
        x_test = rbmup(dbn.rbm{i - 1}, x_test,@sigm);
        dbn.rbm{i} = rbmtrain(dbn.rbm{i},x_train,opts,x_test);
    end
    
end

end
