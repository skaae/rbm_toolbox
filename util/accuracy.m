function [err, other_measures] = accuracy(pred_probs,ey)
%%ACCURACY calculates accuracy
% 
%  INPUTS
%     pred_probs : predicted probabilities for each class
%             ey : one-of-K encoded true classes
%
%   OUTPUT
%            err : accuracy error. The output from error function must be some 
%                  error emasure.
% other_measures : optionally output a struct with other error measures. These 
%                  are not used but stored in opts.val_error_measures and
%                  opts.train_error_measures.
%
%  Copyright (c) Søren Sønderby july 2014

% find predictions and correct labels
[~, pred] = max(pred_probs,[],2);
[~, expected] = max(ey,[],2);

other_measures = {};
err = 1-mean(pred==expected);
end