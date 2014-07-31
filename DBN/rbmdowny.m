function act_vis_y = rbmdowny(rbm,hid_act,prob_or_sample)
%%RBMDOWNY calculates p(v_label = 1 | h) for label units
% This function returns [] for non class rbm's
%
% INPUTS
%   rbm           : A rbm struct
%   hid_act       : the activation of the hidden layer
%   prob_or_sample: 'prob'  : return label probabilities, 
%                   'sample': sample label values based on probabilities 
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
% See also RBMDOWNX RBMUP SAMPLEMATRIX
% 
% Copyright Søren Sønderby June 2014

if rbm.classRBM == 1
      act_vis_y = exp(bsxfun(@plus,rbm.d',hid_act * rbm.U));
      act_vis_y = bsxfun(@rdivide, act_vis_y, sum(act_vis_y, 2));
      if strcmp(prob_or_sample,'sample')
        act_vis_y = samplematrix(act_vis_y);
      end
else
    act_vis_y = [];
end
