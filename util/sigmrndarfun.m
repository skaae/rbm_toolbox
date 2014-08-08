function X = sigmrndarfun(rbm,P)
%%SIGMRND returns neuron value as binary value with activation probability
% X = double(1./(1+exp(-P)))+1*randn(size(P));
X = double(1./(1+exp(-P)) > rbm.rand('double'));
end