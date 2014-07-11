function [ class_vec ] = dbnmakeonehot( dbn,n,k)
%DBNMAKEONEHOT create n one hot encoding of class k given a dbn
    n_classes = size(dbn.rbm{end}.U,2);
    class_vec     = zeros(1,n_classes);
    class_vec(k)  = 1;
    class_vec     = repmat(class_vec,n,1);


end

