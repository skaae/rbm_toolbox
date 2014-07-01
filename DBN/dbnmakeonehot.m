function [ class_vec ] = dbnmakeonehot( dbn,n,k)
%DBNMAKEONEHOT create n one hot encoding of class k given a dbn
    size_last_vis = size(dbn.rbm{end}.W,2);
    n_classes     = size_last_vis- dbn.sizes(end-1);
    class_vec     = zeros(1,n_classes);
    class_vec(k)  = 1;
    class_vec     = repmat(class_vec,n,1);


end

