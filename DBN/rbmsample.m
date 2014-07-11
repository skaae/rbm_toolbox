function [vis_sampled] = rbmsample(rbm,n,k,sampleclass)
%RBMSAMPLE generate n samples from RBM using gibbs sampling with k steps
%   INPUTS:
%       rbm           : a rbm struct
%       n             : number of samples
%       k             : number of gibbs steps before sampling
%       sampleclass   : Class to sample. This is either a scalar giving the
%                       class or a vector of size [n x n_classes] with each 
%                       row corresponding to the one hot encoding of the desired 
%
%   OUTPUTS
%       vis_samples       : samples as a samples-by-n matrix
%
%  NOTES
%   k should quite high. 1000 seems to work for mnist PCD model
%
% Copyright Søren Sønderby June 2014

if nargin == 4   % sample class is given, assume that hintonDBN = 1
    
    % check wether a scalar or a matrix is given
    if isscalar(sampleclass)
        class_vec     = dbnmakeonehot( dbn,n,sampleclass);
    else
        if size(sampleclass,1) ~= n
            error('Given class matrix does not match n');
        end
        class_vec = sampleclass;
    end  
    
else
   class_vec = [];
end


% create n random binary starting vectors based on bias
bx = repmat(rbm.b',n,1);
vis_sampled = double(bx > rand(size(bx)));



% do updown passes k-1 times
for i = 1:k-1
    hid_sampled = rbmup(rbm,vis_sampled,class_vec,@sigmrnd);
    [vis_sampled,~] = rbmdown(rbm,hid_sampled,@sigmrnd);   
    if mod(i,500) == 1
        fprintf('.')
    end
    
end
fprintf('.OK\n')
% in last down pass dont sample binary.
    hid_sampled = rbmup(rbm,vis_sampled,class_vec,@sigmrnd);
    [vis_sampled,~] = rbmdown(rbm,hid_sampled,@sigm);  

end

