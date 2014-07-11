function [vis_sampled] = dbnsample(dbn,n,k,sampleclass)
%%DBNSAMPLE generate n samples from DBN using gibbs sampling with k steps
%   INPUTS:
%       dbn           : a rbm struct
%       n             : number of samples
%       k             : number of gibbs steps befor sampling
%       sampleclass   : Class to sample. This is either a scalar giving the
%                       class or a vector of size [n x n_classes] with each
%                       row corresponding to the one hot encoding of the desired
%                       class
%   OUTPUTS
%       vis_sampled       : samples as a [n x #vis_in_bottom_rbm]
%
%
% Copyright Søren Sønderby June 2014

n_rbm = numel(dbn.rbm);
toprbm = dbn.rbm{end};


%sample top RBM
vis_sampled       = rbmsample(toprbm,n,k,sampleclass);

% deterministicly pass this down the network
for i = (n_rbm - 1):-1:1
    rbm = dbn.rbm{i};
    [vis_sampled,~] = rbmdown(rbm,vis_sampled,@sigm);
end
end

