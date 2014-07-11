function [ rbm ] = rbmapplygrads(rbm,dw,db,dc,du,dd,x,ey,epoch)
%RBMAPPLYGRADS applies momentum and learningrate and updates rbm weights
%   INPUT
%       rbm     : rbm struct
%       opts    : opts struct
%       dw      : w weights change
%       db      : change of bias of visible layer
%       dc      : change of bias of hidden layer
%       du      : change of weights from class labels to hidden layer
%       dd      : chainge of bias in class label hidden layer
%       x       : current minibatch
%       ey      : if hintonDBN one hot encoded class labels otherwise empty
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
%   
% Copyright Søren Sønderby June 2014

% update learningrates
rbm.curMomentum     = rbm.momentum(epoch);
rbm.curLR           = rbm.learningrate(epoch,rbm.curMomentum);

%% l2 regularization
if rbm.L2 >0
 dw = dw -  rbm.L2 * rbm.W;
 if rbm.hintonDBN == 1
    du = du - rbm.L2 * rbm.U;
 end
 
end

%% l1 regularization
if rbm.L1 > 0
    dw =  dw -  rbm.L1 * sign(rbm.W);    %    rbm.W./abs(rbm.W);
    if rbm.hintonDBN == 1
        du = du - rbm.L1 * sign(rbm.U);
    end
end

if rbm.sparsity > 0
    dw = dw - rbm.sparsity;
    if rbm.hintonDBN == 1
        du = du - rbm.sparsity;
    end
end

%% update weights and momentum of regular weights
rbm.vW = rbm.curMomentum * rbm.vW + rbm.curLR * dw; 
rbm.vb = rbm.curMomentum * rbm.vb + rbm.curLR * db;
rbm.vc = rbm.curMomentum * rbm.vc + rbm.curLR * dc;


rbm.W = rbm.W + rbm.vW;
rbm.b = rbm.b + rbm.vb;
rbm.c = rbm.c + rbm.vc;

%% if hintonDBN update weigts and momentum of U and d
if rbm.hintonDBN == 1
    rbm.vU = rbm.curMomentum * rbm.vU + rbm.curLR * du; 
    rbm.vd = rbm.curMomentum * rbm.vd + rbm.curLR * dd;
    rbm.U  = rbm.U + rbm.vU;
    rbm.d  = rbm.d + rbm.vd;
end

%% l2 norm constraint
if rbm.L2norm > 0;
    rbm.W = l2normconstraint( rbm.W,rbm.L2norm );
    if rbm.hintonDBN == 1
        rbm.U = l2normconstraint( rbm.U,rbm.L2norm );
    end

end


end

