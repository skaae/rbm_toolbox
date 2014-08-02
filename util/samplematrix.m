function sample = samplematrix(x)
%SAMPLEMATRIX create a randomized sample from a matrix of probabilities
% assumes that each row of x is normalized. Samples from each of row 
% of x using uniform distribution
% [n_samples,n_classes] = size(x);
% sample = zeros(n_samples,n_classes);
% r = rand(n_samples,1);
% for i = 1:n_samples
%     aux = 0;
%     for j = 1:n_classes
%         aux = aux + x(i,j);
%         if aux >= r(i)
%             sample(i,j) = 1;
%             break;
%         end
%     end
% end
% 

%% vectorized implementation
[n_samples,n_classes] = size(x);
sample = zeros(n_samples,n_classes);

r = rand(n_samples,1);
x_c = cumsum(x,2);
larger = bsxfun(@ge,x_c,r);
[~,idx] = max( larger, [], 2 );

lin_idx = sub2ind(size(x), colon(1,n_samples)', idx);
sample(lin_idx) = 1;


