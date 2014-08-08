function sample = samplematrix(x)
%SAMPLEMATRIX create a randomized sample from a matrix of probabilities
[n_samples,n_classes] = size(x);
sample = zeros(n_samples,n_classes);

r = rand(n_samples,1);
x_c = cumsum(x,2);
larger = bsxfun(@ge,x_c,r);
[~,idx] = max( larger, [], 2 );

lin_idx = sub2ind(size(x), colon(1,n_samples)', idx);
sample(lin_idx) = 1;