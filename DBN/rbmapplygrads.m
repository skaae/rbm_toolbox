function [ rbm ] = rbmapplygrads(rbm,dw,db,dc,epoch)
%RBMAPPLYGRADS applies momentum and learningrate and updates rbm weights
%   INPUT
%       rbm     : rbm struct
%       opts    : opts struct
%       dw      : w weights change
%       db      : change of bias of visible layer
%       dc      : change of bias of hidden layer
%       epoch   : current epoch number 
%   
%   OUTPUT
%       rbm     :  rbm struct with updated weights, LR and momentum
%
%
%  SETTING MOMENTUM
%   Start with a momentum of 0.5. Once the large initial progress in the 
%   reduction of the reconstruction error has settled down to gentle progress, 
%   increase the momentum to 0.9. This shock may cause a transient increase in 
%   the reconstruction error. If this causes a more lasting instability, keep 
%   reducing the learning rate by factors of 2 until the instability disappears.
%
%  SETTING WEIGHT DECAY
%   
% Copyright Søren Sønderby June 2014

% update learningrates
rbm.curMomentum     = rbm.momentum(epoch);
rbm.curLR           = rbm.learningrate(epoch,rbm.curMomentum);

% update momentum and wight change and weight decay


% dw, db,dv are negative gradients
dw = dw -  rbm.L2 * rbm.W;  %apply weight decay
rbm.vW = rbm.curMomentum * rbm.vW + rbm.curLR * dw ; 
rbm.vb = rbm.curMomentum * rbm.vb + rbm.curLR * db;
rbm.vc = rbm.curMomentum * rbm.vc + rbm.curLR * dc;

% update weights

rbm.W = rbm.W + rbm.vW;
rbm.b = rbm.b + rbm.vb;
rbm.c = rbm.c + rbm.vc;


end

