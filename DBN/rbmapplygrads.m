function [ rbm ] = rbmapplygrads(rbm,dw,db,dc,x,epoch)
%RBMAPPLYGRADS applies momentum and learningrate and updates rbm weights
%   INPUT
%       rbm     : rbm struct
%       opts    : opts struct
%       dw      : w weights change
%       db      : change of bias of visible layer
%       dc      : change of bias of hidden layer
%       x       : current minibatch
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

%% l2 regularization
if rbm.L2 >0
 dw = dw -  rbm.L2 * rbm.W;
end

%% l1 regularization
if rbm.L1 > 0
    dw =  dw -  rbm.L1 *sign(rbm.W)    %    rbm.W./abs(rbm.W);
end

%% l2 norm constraint
if rbm.L2norm > 0;
    input = sum(rbm.W.^2,2);
    norm_const = sqrt(input/rbm.L2norm);        % normalization constant
    norm_const(norm_const < 1) = 1;             % find units below threshold
    rbm.W = bsxfun(@rdivide,rbm.W,norm_const);  %rescale weights above threshold
end

% dw, db,dv are negative gradients
rbm.vW = rbm.curMomentum * rbm.vW + rbm.curLR * (dw); 
rbm.vb = rbm.curMomentum * rbm.vb + rbm.curLR * db;
rbm.vc = rbm.curMomentum * rbm.vc + rbm.curLR * dc;

% update weights

rbm.W = rbm.W + rbm.vW;
rbm.b = rbm.b + rbm.vb;
rbm.c = rbm.c + rbm.vc;


end

