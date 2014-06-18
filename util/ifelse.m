function result = ifelse(condition,trueResult,falseResult)
%IFELSE oneline if else.
% IFELSE(condition,trueResult,falseResult)  
    error(nargchk(3,3,nargin));
  if condition
    result = trueResult;
  else
    result = falseResult;
  end