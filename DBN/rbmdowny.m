function act_vis_y = rbmdowny(rbm,hid_act,act_func)
%%RBMDOWNY calculates p(v_label = 1 | h) for label units
% This function returns [] for non rbm's
%
% INPUTS
%   rbm           : A rbm struct
%   hid_act          : the activation of the hidden layer
%   act_func      : the activation function, @sigm | @sigmrnd
%
% OUTPUTS
%   act_vis_y : The activation of the class label visible units
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
% Copyright Søren Sønderby June 2014

if rbm.classRBM == 1
%     vis_label_bias = repmat(rbm.d', size(hid, 1), 1);
%     act_vis_label = act_func(vis_label_bias + hid * rbm.U);
      act_vis_y = act_func(bsxfun(@plus,rbm.d',hid_act * rbm.U));
else
    act_vis_y = [];
end
