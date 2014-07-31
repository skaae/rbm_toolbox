function [ p ] = predict( x )
%PREDICT find most likely class i.e idx of max value in each row
[~,p] =  max(x,[],2);

end

