function rbm = rbmoverfitting( rbm,x_train,val_samples,opts,epoch)
%RBMOVERFITTING function for monitoring overfitting,
%   measures the free energy a validation set and a training  set and retunrs
%   the ratio. If the ratio  free_energy_val / free_energy_train is much
%   less than 1 we are possibly overfitting.
%   See A practical guide to training restricted Boltzmann machines
%   section 6.
%
%  INPUTS
%         rbm : a rbm struct
%     x_train : x_training samples
% val_samples : selected validation samples
%        opts : opts struct
%       epoch : current epoch
%
%  OUTPUTS
%   rbm       : a rbm struct with updated ratioy and ratiox
% Copyright Søren Sønderby july 2014
e_val = rbmfreeenergy(rbm,opts.x_val,opts.y_val);
e_train = rbmfreeenergy(rbm,x_train(val_samples,:),...
    opts.y_train(val_samples,:));
rbm.ratioy(end+1) = e_val /e_train;;
rbm.ratiox(end+1) = epoch;
end

