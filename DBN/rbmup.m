function x = rbmup(rbm, x,act)
%%RBMUP calculate p(h=1|v)
% see "A practical guide to training restricted Boltzmann machines" eqn 7
% act is the activation function. Currently either sigm or sigmrnd
    x = act(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
end
