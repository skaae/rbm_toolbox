function [ W ] = l2normconstraint( W,L2norm )
%L2NORMCONSTRAINT applies L2normconstraint to a weight matrix
%   INPUT
%            W : weight matrix    
%       L2norm : double specifying constraint on the incoming weight sizes
%                to each nuron. If the L2norm is above this value the
%                weights for this neuron is rescaled to L2norm. See 
%                http://arxiv.org/abs/1207.0580
%
%   OUTPUT
%           W :  rescaled weights below threshold
%
% Copyright (c) Søren Sønderby july 2014
input = sum(W.^2,2);
norm_const = sqrt(input/L2norm);    % normalization constant
norm_const(norm_const < 1) = 1;     % find units below threshold
W = bsxfun(@rdivide,W,norm_const);  %rescale weights above threshold

end

