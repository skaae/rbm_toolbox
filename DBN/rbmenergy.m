function energy = rbmenergy(rbm,x)
%RBMENERGY calculates the energy of a RBM given some data 
% 
%  NOTATION:
%    w  : weights
%    b  : bias of visible layer
%    c  : bias of hidden layer



% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.

% The energy is calculated as: 
%       E &=& -\sum^{}_{i}s_ib_i - \sum^{}_{i<j}s_is_j w_{ij}
%
% We have no biases so we omit tye first sum.  

% for i = 1:n_samples
%     hid = hidden_state(:,i);
%     vis = visible_state(:,i);
%     
%     energy = energy -hid'*rbm_w*vis; 
%     
% end
% energy = energy/n_samples;


hid = rbmup(rbm,x,@sigm);
vis = x;
n_samples = size(x,1);
energy = (trace(-hid*rbm.W*vis') + sum(vis*rbm.b)) / n_samples;
end
