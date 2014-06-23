function x = rbmdown(rbm, x,act)
%%RBMDOWN calculates p(v = 1 | h)
% see "A practical guide to training restricted Boltzmann machines" eqn 8
% act is the activation function. currently either sigm or sigmrnd
% Modified by Søren Sønderby June 2014
    x = act(repmat(rbm.b', size(x, 1), 1) + x * rbm.W);
end
