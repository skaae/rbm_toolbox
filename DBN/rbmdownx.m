function act_vis_x = rbmdownx(rbm,hid_act,act_func)
%%RBMDOWNX calculates p(v = 1 | h) for non label units
%
% INPUTS
%   rbm           : A rbm struct
%   hid_act          : the activation of the hidden layer
%   act_func      : the activation function, @sigm | @sigmrnd
%
% OUTPUTS
%   act_vis_x     : The activation of the x visible units
%
% see "A practical guide to training restricted Boltzmann machines" eqn 8
% act is the activation function. currently either sigm or sigmrnd
%
% NOTATION
% data  : all data given as      [n_samples   x #vis]
%    v  : all data given as      [n_samples   x #vis]
%   ey  : all data given as      [n_samples   x #n_classes]
%    W  : vis - hid weights      [ #hid       x #vis ]
%    U  : label - hid weights    [ #hid       x #n_classes ]
%    b  : bias of visible layer  [ #vis       x 1]
%    c  : bias of hidden layer   [ #hid       x 1]
%    d  : bias of label layer    [ #n_classes x 1]
%
% Modified by Søren Sønderby June 2014

% rep vis bias n_samples times then calculate act of vis from hid
%vis_x_bias = repmat(rbm.b', size(hid, 1), 1);
%act_vis_x = act_func(vis_x_bias + hid * rbm.W);  % activation of visible units
act_vis_x = act_func( bsxfun(@plus,rbm.b',hid_act * rbm.W) );
end
