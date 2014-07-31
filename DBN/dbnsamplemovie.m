function [] = dbnsamplemovie(dbn,n,k,fout,samplefreq,visualizer,sampleclass)
%%DBNSAMPLEMOVIE generates movie of sampling process from DBN
%
%   INPUTS:
%       dbn               : a rbm struct
%       n                 : number of samples 
%       k                 : number of gibbs steps.
%       fout              : location of output movie
%       samplefreq        : samples between a picture is captured
%       visualizer        : a function which returns a plot
%       sampleclass       : Class to sample. This is either a scalar giving the
%                           class or a vector of size [n x n_classes] with each
%                           row corresponding to the one hot encoding of the desired
%                           class to be sampled.
%
% See also DBNSAMPLE
%
% Copyright Søren Sønderby June 2014
n_rbm = numel(dbn.rbm);

if nargin == 7   % sample class is given, assume that classRBM = 1 
    % check wether a scalar or a matrix is given
    if isscalar(sampleclass)
        class_vec     = dbnmakeonehot( dbn,n,sampleclass);
    else
        if size(sampleclass,1) ~= n
            error('Given class matrix does not match n');
        end
        class_vec = sampleclass;
    end 
else
    class_vec = [];
end

% create starting state from bias
toprbm = dbn.rbm{end};
bx = repmat(toprbm.b',n,1);
vis_sampled = double(bx > rand(size(bx)));

% create movie
close all
figure;
vidObj = VideoWriter(fout);
vidObj.Quality = 100;
vidObj.FrameRate = 30;
open(vidObj);

for i = 1:k
    hid_sampled = rbmup(toprbm,vis_sampled,class_vec,@sigmrnd);
    vis_sampled = rbmdownx(toprbm,hid_sampled,@sigm);
    
    if mod(i-1,samplefreq) == 0
        samples = vis_sampled;
        
        if n_rbm ~= 1 % if non-single DBN sample values before pass down
            samples = double(samples > rand(size(samples)));
        end
        
        % pass to bottom layer
        for j = (n_rbm - 1):-1:1
            rbm = dbn.rbm{j};
            samples = rbmdownx(rbm,samples,@sigm);
        end
        
        fprintf('Gibbs steps: %i\n',i)
        visualizer(samples');
        axis equal
        writeVideo(vidObj, getframe(gca));
    end
    vis_sampled = double(vis_sampled > rand(size(vis_sampled))); 
end






