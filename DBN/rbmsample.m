function [vis_sampled] = rbmsample(rbm,n,k,sampleclass)
%RBMSAMPLE generate n samples from RBM using gibbs sampling with k steps
%   INPUTS:
%       rbm               : a rbm struct
%       n                 : number of samples
%       k                 : number of gibbs steps
%       sampleclas s      : optional class to sample, encoded as oneHot
%   OUTPUTS
%       vis_samples       : samples as a samples-by-n matrix
%
%  NOTES
%   k should quite high. 1000 seems to work for mnist PCD model
% 
% Copyright Søren Sønderby June 2014

% create n random binary starting vectors based on bias
bx = repmat(rbm.b',n,1);
vis_sampled = double(bx > rand(size(bx)));
if nargin == 4  % sample class is given, assume that hintonDBN = 1
    vis_sampled = [vis_sampled repmat(sampleclass,n,1)];
end

% do updown passes k-1 times
for i = 1:k-1
    hid_sampled = rbmup(rbm,vis_sampled,@sigmrnd);
    vis_sampled = rbmdown(rbm,hid_sampled,@sigmrnd);
    
end

% in last down pass dont sample binary.
hid_sampled = rbmup(rbm,vis_sampled,@sigmrnd);
vis_sampled = rbmdown(rbm,hid_sampled,@sigm);

end

