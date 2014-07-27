function sample = randsample(x)
%RANDSAMPLE create a randomized sample from a matrix of probabilities
% assumes that each row of x is normalized. Samples from each of row 
% of x using uniform distribution
[n_samples,n_classes] = size(x);
sample = zeros(n_samples,n_classes);
r = rand(1,n_samples);
for i = 1:n_samples
    aux = 0;
    for j = 1:n_classes
        aux = aux + x(i,j);
        if aux >= r(i)
            sample(i,j) = 1;
            break;
        end
    end
end

