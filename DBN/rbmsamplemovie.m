function rbmsamplemovie(rbm,n,k,fout,samplefreq,visualizer)
%RBMSAMPLEMOVIE generate n samples from RBM using gibbs sampling with k steps
%   INPUTS:
%       rbm               : a rbm struct
%       n                 : number of samples
%       k                 : number of gibbs steps 
%       samplefreq        : samples between a picture is captured
%       visualizer        : a function which returns a plot
%   OUTPUTS
%       vis_samples       : samples as a samples-by-n matrix
%
%  NOTES
%   k should quite high. 1000 seems to work for mnist PCD model
%   The following set up will give you a model that can generate digits:
%   ------CODE------
%         load mnist_uint8;
%         train_x = double(train_x) / 255;
%         test_x  = double(test_x)  / 255;
%         rand('state',0)
%         dbn.sizes = [500];
%         opts.traintype = 'PCD';
%         opts.numepochs =   100; % probably way to high?
%         opts.batchsize = 100;
%         opts.cdn = 1; % contrastive divergence
%         T = 50;       % momentum ramp up
%         p_f = 0.9;    % final momentum
%         p_i = 0.5;    % initial momentum
%         eps = 0.05;    % initial learning rate
%         f = 0.95;     % learning rate decay
%         opts.learningrate = @(t,momentum) eps.*f.^t*(1-momentum); 
%         opts.momentum     = @(t) ifelse(t < T, p_i*(1-t/T)+(t/T)*p_f,p_f);
%         opts.L2 = 0.00;
%         dbn = dbnsetup(dbn, train_x, opts);
%         dbn = dbntrain(dbn, train_x, opts,test_x);
%         rbm = dbn.rbm{1};
%         digits = rbmsample(rbm,50,10000);
%         visualize(digits')
% Copyright Søren Sønderby June 2014

% create n random binary starting vectors based on bias
bx = repmat(rbm.b',n,1);
vis_sampled = double(bx > rand(size(bx)));

close all
figure;
nFrames = floor(k/10);
vidObj = VideoWriter(fout);
vidObj.Quality = 100;
vidObj.FrameRate = 10;
open(vidObj);


frame = 0;
for i = 1:k
    hid_sampled = rbmup(rbm,vis_sampled,@sigmrnd);
    vis_sampled = rbmdown(rbm,hid_sampled,@sigmrnd);
    
    if mod(i-1,samplefreq) == 0
        frame = frame+1;
        fprintf('Gibbs steps: %i\n',i)
        digits = rbmdown(rbm,hid_sampled,@sigm);
        visualizer(digits');
        axis equal
        writeVideo(vidObj, getframe(gca));
    end
    
end
    hid_sampled = rbmup(rbm,vis_sampled,@sigmrnd);
    vis_sampled = rbmdown(rbm,hid_sampled,@sigm);
end

