function sample = samplevec(rbm,x)
%SAMPLEVEC create a randomized sample from a vector of probabilities
r = rbm.rand(1);
x_c = cumsum(x,2);
larger = bsxfun(@ge,x_c,r);
sample = double(bsxfun(@eq,1,cumsum(larger,2)));


