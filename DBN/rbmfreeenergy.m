function freeenergy = rbmfreeenergy(rbm,v,ey)
%RBMFREEENERGY calculates the free energy of a RBM
% NOT WORKING ATM
%
% NOTATION:
% w : weights
% b : bias of visible layer
% c : bias of hidden layer
%
%
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


[n_hidden, ~] = size(rbm.W);
n_samples = size(v,1);

% im not sure but the definition of free energy for a classification RBM
% and a normal RBM seems to be different. See
% http://deeplearning.net/tutorial/rbm.html
% (eq 9) for a normal RBM and "Learning Algorithms for the Classification RBM
% the eq. showing derivation of p(y|x)


    
freeenergy = 0; % acuumulator over samples
for t = 1:n_samples
    x_t = v(t,:);
    if rbm.classRBM == 1
        y_idx = find(ey(t,:));
        res = -rbm.d(y_idx); % accumulator over hidden
    else
        res = - x_t*rbm.b;
    end
    
    for j = 1:n_hidden
        if rbm.classRBM == 1
            res = res - softplus(rbm.c(j)+rbm.W(j,:)*x_t'+rbm.U(j,y_idx));
        else
            res = res - softplus(rbm.c(j)+rbm.W(j,:)*x_t' );
        end
         
    end
    freeenergy = freeenergy + res/n_samples; % acumulate
end

end