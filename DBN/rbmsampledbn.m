function [samples] = rbmsampledbn(dbn,n,k)
%RBMSAMPLEDBN generates samples from a dbn
%   INPUTS:
%       dbn               : a rbm struct
%       n                 : number of samples
%       k                 : number of gibbs steps 
%   OUTPUTS
%       vis_samples       : samples as a samples-by-n matrix
%
% Copyright Søren Sønderby June 2014

n_rbm = numel(dbn.rbm);


% create n random binary starting vectors based on bias
toprbm = dbn.rbm{end};
samples = rbmsample(toprbm,n,k);

% deterministicly pass this down the network
for i = (n_rbm - 1):-1:1
    rbm = dbn.rbm{i};
    samples = rbmdown(rbm,samples,@sigm);
    
end


end

