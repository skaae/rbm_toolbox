function [v2,h1,h2] = cdn(rbm,v1,n )
%CDN performs contrastive divergence n_times
%   Contrastive divergence sampling.
%   see "A practical guide to training restricted Boltzmann machines" section
%   3.4.
%   INPUTS:
%     rbm: a rbm struct
%     v1 ; the initial state of the hidden units
%     n  :  number of contrastive divergence steps
%   OUTPUTS
%     v2 : Reconstructed visible state (for negative phase)
%     h1 : First reconstructed hidden state (for positive phase)
%     h2 : Final reconstructed hidden state, not bernoulli sampled (for neg phase)
%    
% See also
%   Hinton, G. (2002). Training Products of Experts by Minimizing Contrastive 
%   Divergence. Neural Compu- tation, 14, 1771?1800.
h1 = rbmup(rbm,v1,@sigmrnd);  % keep

hid_rnd = h1;
for i=1:n
    vis_sigm = rbmdown(rbm,hid_rnd,@sigmrnd);
    hid_sigm = rbmup(rbm,vis_sigm,@sigm);
    hid_rnd  = double(hid_sigm>rand(size(hid_sigm)));  %sample
end
v2 = vis_sigm;
h2 = hid_sigm;
end

