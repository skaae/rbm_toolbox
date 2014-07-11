function act_hid = rbmup(rbm, vis,ey,act_func)
%%RBMUP calculate p(h=1|v)
% INPUTS
%   rbm        : A rbm struct
%   vis        : the activation of the visible layer
%   ey         : if hintonDBN the activation of the class labels
%   act_func   : the activation function, either @sigm or @sigmrnd
%
% OUTPUTS
%   act_of_hid : the activation of the hidden layer
%
% see "A practical guide to training restricted Boltzmann machines" eqn 7
% act is the activation function. Currently either sigm or sigmrnd
%
% 
% NOTATION
%    v  : all data given as      [n_samples   x #vis]
%   ey  : all data given as      [n_samples   x #n_classes]
%    W  : vis - hid weights      [ #hid       x #vis ]
%    U  : label - hid weights    [ #hid       x #n_classes ]
%    b  : bias of visible layer  [ #vis       x 1]
%    c  : bias of hidden layer   [ #hid       x 1]
%    d  : bias of label layer    [ #n_classes x 1]
% Copyright Søren Sønderby June 2014

% repmat hid bias n_samples times then calc activation from visible units. 
hid_bias = repmat(rbm.c', size(vis, 1), 1);
act_hid = hid_bias + vis * rbm.W';        %activation of from visible units

% for the hinton DBN add the activation coming from the label visible units
if rbm.hintonDBN == 1
    act_hid  = act_hid +  ey * rbm.U'; 
end
act_hid = act_func(act_hid); %apply activation function
