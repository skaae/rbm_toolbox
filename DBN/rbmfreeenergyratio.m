function ratio = rbmfreeenergyratio(rbm,x,x_val)
%RBMFREEENERBYRATIO calculates the free energy ratio between x anx x_val
% see https://dl.dropboxusercontent.com/u/19557502/5_03_free_energy.pdf
%
% NOTATION
% data : all data given as [n_samples x #vis]
% v : all data given as [n_samples x #vis]
% ey : all data given as [n_samples x #n_classes]
% W : vis - hid weights [ #hid x #vis ]
% U : label - hid weights [ #hid x #n_classes ]
% b : bias of visible layer [ #vis x 1]
% c : bias of hidden layer [ #hid x 1]
% d : bias of label layer [ #n_classes x 1]
% Copyright Søren Sønderby july 2014

assert(size(x,1)==size(x_val,1))
if rbm.classRBM==1
    error('not implemented for class RBM`s')
end

F_x     = freeenergy(rbm,x);
F_x_val = freeenergy(rbm,x_val);
ratio   = mean(F_x_val ./ F_x);

    function F = freeenergy(rbm,x)
        % calculates free energy for all samples in x
        wxc = softplus(bsxfun(@plus,rbm.c,rbm.W*x'));
        F =-(rbm.b'*x' + sum(wxc,1));
    end
end