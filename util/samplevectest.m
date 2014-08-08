function sample = samplevectest(rbm,x)
%SAMPLEVEC create a randomized sample from a vector of probabilities
n_classes = size(x,2);
sample = zeros(1,n_classes);
r = rbm.rand('double');
x_c = cumsum(x,2);
larger = bsxfun(@ge,x_c,r);
[~,idx] = max( larger, [], 2 );
sample(idx) = 1;