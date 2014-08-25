function res = normalizeexprowvec(x)
%NORMALIZEEXP numeically stable calculation of exp(X) / sum(exp(x))
% SEE http://timvieira.github.io/blog/tag/numerical.html
N = size(x,2);
exp_x = exp(x-max(x,[],2));
res = exp_x / (exp_x*ones(N,1));
end