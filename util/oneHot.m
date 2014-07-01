function oneHotLabels = oneHot(labels)

% Takes a vector of size n by 1 as input and creates a one-hot encoding of its
% elements.

valueLabels = unique(labels);
valueLabels = valueLabels(~isnan(valueLabels));
nLabels = numel(valueLabels);
nSamples = numel(labels);

oneHotLabels = zeros(nSamples, nLabels);

for i = 1:nLabels
	oneHotLabels(:,i) = (labels == valueLabels(i));
end

oneHotLabels = oneHotLabels*2-1;