function [] = rbmsampledbnmovie(dbn,n,k,fout,samplefreq,visualizer)
%RBMSAMPLEDBNMOVIE generates movie of sampling from DBN
%   INPUTS:
%       dbn               : a rbm struct
%       n                 : number of samples
%       k                 : number of gibbs steps
%       fout              : location of output movie
%       samplefreq        : samples between a picture is captured
%       visualizer        : a function which returns a plot
%
% Copyright Søren Sønderby June 2014

n_rbm = numel(dbn.rbm);

if n_rbm == 1
    % todo merge this code into this functino
    rbmsamplemovie(dbn.rbm{1},n,k,fout,samplefreq,visualizer)
else
    
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
    
    for i = 1:k
        hid_sampled = rbmup(toprbm,vis_sampled,@sigmrnd);
        vis_sampled = rbmdown(toprbm,hid_sampled,@sigmrnd);
        
        
        if mod(i-1,samplefreq) == 0
            samples = vis_sampled;
            for j = (n_rbm - 1):-1:1
                rbm = dbn.rbm{j};
                samples = rbmdown(rbm,samples,@sigm);
                
            end
            
            fprintf('Gibbs steps: %i\n',i)
            visualizer(samples');
            axis equal
            writeVideo(vidObj, getframe(gca));
            
        end
        
    end
end





