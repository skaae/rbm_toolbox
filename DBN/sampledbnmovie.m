function [] = sampledbnmovie(dbn,n,k,fout,samplefreq,visualizer,sampleclass)
%%SAMPLEDBNMOVIE generates movie of sampling from DBN
%   INPUTS:
%       dbn               : a rbm struct
%       n                 : number of samples
%       k                 : number of gibbs steps
%       fout              : location of output movie
%       samplefreq        : samples between a picture is captured
%       visualizer        : a function which returns a plot
%       sampleclass       : class to sample if hintonDBN, an integer
% Copyright Søren Sønderby June 2014

n_rbm = numel(dbn.rbm);


if nargin == 7  % sample class is given, assume that hintonDBN = 1
    class_vec = dbnmakeonehot( dbn,n,sampleclass);
else
    class_vec = [];
end



%% create movie
close all
figure;
vidObj = VideoWriter(fout);
vidObj.Quality = 100;
vidObj.FrameRate = 10;
open(vidObj);



% sample from top rbm
toprbm = dbn.rbm{end};

bx = repmat(toprbm.b',n,1);
vis_sampled = double(bx > rand(size(bx)));
vis_sampled = [vis_sampled class_vec];


for i = 1:k
    hid_sampled = rbmup(toprbm,vis_sampled,@sigmrnd);
    vis_sampled = rbmdown(toprbm,hid_sampled,@sigm);
    
    if mod(i-1,samplefreq) == 0
        samples = vis_sampled;
        % in the hintonDBN nclasses was augmented to the visible layer of
        % the last RBM, remove these before down pass...
        if nargin == 7
            samples = samples(:,1:dbn.sizes(end-1));
        end
        
        if n_rbm ~= 1 % if non-single DBN sample values before pass down
            samples = double(samples > rand(size(samples)));
        end
            
        
        for j = (n_rbm - 1):-1:1
            rbm = dbn.rbm{j};
            samples = rbmdown(rbm,samples,@sigm);    
        end
 
        fprintf('Gibbs steps: %i\n',i)
        visualizer(samples');
        axis equal
        writeVideo(vidObj, getframe(gca));    
    end
    vis_sampled = double(vis_sampled > rand(size(vis_sampled)));
    
end






