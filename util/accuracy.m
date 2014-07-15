function [acc_err, other_measures] = accuracy(confusion)
%% accuracy calculates accuracy
% http://bioinformatics.oxfordjournals.org/content/16/5/412.full.pdf+html
% The confusino matrix must be 2 x 2 in the following format:
% preduction
% ____________________________
% | pos neg
% __________|_________________
% true | pos| 1,1 TP | 1,2 FN
% clas | neg| 2,1 FP | 2,2 TN
n_classes = size(confusion,3);
acc = zeros(1,n_classes);
for i = 1:n_classes
    TP = confusion(1,1,i);
    FN = confusion(1,2,i);
    FP = confusion(2,1,i);
    TN = confusion(2,2,i);
    acc(i) = ((TP + TN) / (TP + TN + FP + FN));
end
acc_err = 1-mean(acc);
other_measures = struct();
end