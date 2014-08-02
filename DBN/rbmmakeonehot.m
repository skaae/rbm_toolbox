function [ class_vec ] = rbmmakeonehot( rbm,n,k)
%DBNMAKEONEHOT create n one hot encoding of class k given a dbn
%
% Copyright Søren Sønderby july 2014
n_classes = size(rbm.U,2);
class_vec     = zeros(n,n_classes);
class_vec(:,k)  = 1;
end

