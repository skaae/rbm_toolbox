function sample = samplematrixtest(x)
%SAMPLEMATRIX create a randomized sample from a matrix of probabilities
% assumes that each row of x is normalized. Samples from each of row 
% of x using uniform distribution
[n_samples,n_classes] = size(x);

r = rand(n_samples,1);

x_c = cumsum(x,2);
larger = bsxfun(@gt,x_c,r);
[~,idx] = max( larger, [], 2 );

lin_idx = sub2ind(size(x), colon(1,n_samples)', idx);

sample = zeros(n_samples,n_classes);
sample(lin_idx) = 1;