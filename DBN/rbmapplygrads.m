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
% Copyright Søren Sønderby June 2014

% update learningrates
rbm.curMomentum     = rbm.momentum(epoch);
rbm.curLR           = rbm.learningrate(epoch,rbm.curMomentum);

% update momentum and wight change
rbm.vW = rbm.curMomentum * rbm.vW + rbm.curLR * dw; 
rbm.vb = rbm.curMomentum * rbm.vb + rbm.curLR * db;
rbm.vc = rbm.curMomentum * rbm.vc + rbm.curLR * dc;

% update weights
rbm.W = rbm.W + rbm.vW;
rbm.b = rbm.b + rbm.vb;
rbm.c = rbm.c + rbm.vc;


end

