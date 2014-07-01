function [samples] = dbnsample(dbn,n,k,sampleclass)
%%DBNSAMPLE generates samples from a dbn
%   INPUTS:
%       dbn               : a rbm struct
%       n                 : number of samples
%       k                 : number of gibbs steps
%       sampleclass       : class to sample if hintonDBN, an integer
%   OUTPUTS
%       vis_samples       : samples as a samples-by-n matrix
%
%   EXAMPLE
        %   The following set up will give you a model that can generate digits:
%   ------CODE------
%         load mnist_uint8;
%         train_x = double(train_x) / 255;
%         test_x  = double(test_x)  / 255;
%         opts = dbncreateopts();
%         dbn = dbnsetup(dbn, train_x, opts);
%         dbn = dbntrain(dbn, train_x, opts);
%         digits = dbnsample(dbn,50,1000);
%         visualize(digits')
%
% Copyright Søren Sønderby June 2014

n_rbm = numel(dbn.rbm);
toprbm = dbn.rbm{end};

if nargin == 4   % sample class is given, assume that hintonDBN = 1
    class_vec     = dbnmakeonehot( dbn,n,sampleclass);
    samples       = rbmsample(toprbm,n,k,class_vec);
    
    % the samples are n_classes too big to pass through the remaining layers
    samples = samples(:,1:dbn.sizes(end-1));
else
    samples = rbmsample(toprbm,n,k);
end


% deterministicly pass this down the network
for i = (n_rbm - 1):-1:1
    rbm = dbn.rbm{i};
    samples = rbmdown(rbm,samples,@sigm);  
end


end

