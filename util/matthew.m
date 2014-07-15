function [mcc_err, other_measures] = matthew(confusion)
%MATTHEW  calculates matthew correlation
% http://bioinformatics.oxfordjournals.org/content/16/5/412.full.pdf+html
% The confusino matrix must be 2 x 2 in the following format:
% preduction
% ____________________________
% | pos neg
% __________|_________________
% true | pos| 1,1 TP | 1,2 FN
% clas | neg| 2,1 FP | 2,2 TN
n_classes = size(confusion,3);
mcc = zeros(1,n_classes);
for i = 1:n_classes
    TP = confusion(1,1,i);
    FN = confusion(1,2,i);
    FP = confusion(2,1,i);
    TN = confusion(2,2,i);
    mcc_denom = (TP+FP) * (TP+FN) * (TN+FP) * (TN + FN);
    
    %maybe check if mcc_denom is nan??? set it to one if so?
    % "If any of the four sums in the denominator is zero, the denominator can 
    % be arbitrarily set to one; this results in a Matthews correlation 
    % coefficient of zero, which can be shown to be the correct limiting value."
    mcc(i) = (TP * TN - FP * FN) ./ sqrt(mcc_denom);
    if isnan(mcc(i))
        mcc(i) = 0;
    end
    
end
mcc_err = 1-mean(mcc);
other_measures = struct();
end