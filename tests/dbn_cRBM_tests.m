rng('default');rng(1);

%% init test weights 
x = rand(100,1);
y = 2;
y_vec = [0 0 1 0]';
W = rand(20,100)-0.5;
W = W ./ max(size(W));
U = rand(20,4)-0.5;
U = U ./ max(size(U));
d = zeros(4,1);
c = zeros(20,1);
b = zeros(100,1);


% init RBM with same weights
rbm.classRBM = 1;
rbm.dropout_hidden = 0;
rbm.W = W;
rbm.U = U;
rbm.d = d;
rbm.c = c;
rbm.b = b;
x_rbm = x';
rbm.rbmdowny = @rbmdownyclassrbm;
rbm.zeros = @zeros;
rbm.rand = @rand;
opts.y_train = y_vec';
opts.traintype = 'CD';
opts.cdn = 1;
opts.batchsize = 1;


%[opts, valid_fields] = dbncreateopts();

%opts.classRBM = 1;
%dbn = dbnsetup([20], x_rbm, opts);




%% Generating the statistics for CD-1 (generative) learning

%%%%%%%%%%%%%%%%%%%%%%%%%
% Check subfunctions    %
%%%%%%%%%%%%%%%%%%%%%%%%%


% p(h | y, x)
p_h_given_y_x  = 1 ./ (1 + exp(-(W*x + c + U*y_vec)));
rng('default');rng(2);
h_pos = rand(20,1) < p_h_given_y_x;

% p(y | h)
p_y_given_h_non_norm= exp( U'*h_pos + d );
p_y_given_h = p_y_given_h_non_norm / sum(p_y_given_h_non_norm);

%% Test RBMUP and RBMDOWNY
my_p_h_given_y_x = rbmup(rbm, x_rbm,opts.y_train,@sigm);
rng('default');rng(2);
my_h_pos = rbmup(rbm, x_rbm,opts.y_train,@sigmrnd);
my_p_y_given_h = rbm.rbmdowny(rbm,h_pos');

assert(all(p_h_given_y_x == my_p_h_given_y_x'));
assert(all(p_y_given_h == my_p_y_given_h'));
assert(all(h_pos == my_h_pos'));



%% Sample from p(y | h)
y_sample = zeros(4,1);
sumsofar = 0;
rng('default');rng(1);
r=rand();
done=0;
for y_=0:3  %iterate over classes
  sumsofar = sumsofar + p_y_given_h(y_+1);
  if sumsofar >= r & done==0
    y_sample(y_+1) = 1;
    done = 1;
  end
end
y_vec_neg = y_sample;


%% check samplematrix and rbmdowny with sample
rng('default');rng(1);
my_y_vec_neg1 = samplematrix(p_y_given_h');
rng('default');rng(1);
my_y_vec_neg2 = rbm.rbmdowny(rbm,h_pos');
my_y_vec_neg2 = samplematrix(my_y_vec_neg2);


assert(all(y_vec_neg == my_y_vec_neg1'));
assert(all(y_vec_neg == my_y_vec_neg2'))

% p(x | h)
p_x_given_h = 1 ./ (1 + exp(-(W'*h_pos + b)));
rng('default');rng(1);
x_neg = rand(100,1) < p_x_given_h;

%% check rbmdownx
my_p_x_given_h = rbmdownx(rbm,h_pos',@sigm);
rng('default');rng(1);
my_x_neg = rbmdownx(rbm,h_pos',@sigmrnd);
assert(all(p_x_given_h == my_p_x_given_h'));
assert(all(x_neg == my_x_neg'));

% p(h | y, x)
p_h_given_y_x_neg  = 1 ./ (1 + exp(-(W*x_neg + c + U*y_vec_neg)));
rng('default');rng(1);
h_neg = rand(20,1) < p_h_given_y_x_neg;

%% check RBMUP
my_p_h_given_y_x_neg = rbmup(rbm,x_neg',y_vec_neg',@sigm);
rng('default');rng(1);
my_h_neg =  rbmup(rbm,x_neg',y_vec_neg',@sigmrnd);
assert(all(p_h_given_y_x_neg == my_p_h_given_y_x_neg'));
assert(all(h_neg == my_h_neg'))

% p(y | x)
rng('default');rng(1);
Wx = W*x;
p_y_given_x = zeros(4,1);
for y_ = 0:3
  p_y_given_x(y_+1,1) = exp( d(y_+1,1) + sum(log(1 + exp( Wx + c + U(:,y_+1)))));
end
p_y_given_x = p_y_given_x / sum(p_y_given_x);

%% check rbmpygivenx
rng('default');rng(1);
my_p_y_given_x = rbmpygivenx(rbm,x','test');
assert(all(abs(p_y_given_x - my_p_y_given_x') < 10^-15))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check rbmgenerative and rbmdiscriminative    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Computing the gradients for discriminative learning
% check the gradients by running all the previous calculations using a same
% random seed
%
%  1) p(h | y, x) and sample hidden            [h_pos       = rbmup(rbm,x,ey,@sigmrnd)]
%  2) p(y|h) calculate class probs and sample  [y_vec_neg = rbmdowny(rbm,h_pos','sample')]
%  3) p(x|h)                                   [x_neg       = rbmdownx(rbm,h_pos',@sigmrnd)]
%  4) p(h | y, x)                              [h_neg = rbmup(rbm,x_neg',y_vec_neg',@sigmrnd)]
%  5) p(y|x)                                   [p_y_given_x =  rbmpygivenx(rbm,x','test')]

clear h_pos y_vec_neg x_neg h_neg p_y_given_x
rng('default');rng(1);

% p(h | y, x)
p_h_given_y_x  = 1 ./ (1 + exp(-(W*x + c + U*y_vec)));
h_pos = rand(20,1) < p_h_given_y_x;

% p(y | h)
p_y_given_h_non_norm= exp( U'*h_pos + d );
p_y_given_h = p_y_given_h_non_norm / sum(p_y_given_h_non_norm);

% p(x | h)
p_x_given_h = 1 ./ (1 + exp(-(W'*h_pos + b)));
x_neg = rand(100,1) < p_x_given_h;

%% Sample from p(y | h)
y_sample = zeros(4,1);
sumsofar = 0;
r=rand();
done=0;
for y_=0:3  %iterate over classes
  sumsofar = sumsofar + p_y_given_h(y_+1);
  if sumsofar >= r & done==0
    y_sample(y_+1) = 1;
    done = 1;
  end
end
y_vec_neg = y_sample;


% p(h | y, x)
p_h_given_y_x_neg  = 1 ./ (1 + exp(-(W*x_neg + c + U*y_vec_neg)));
h_neg = rand(20,1) < p_h_given_y_x_neg;


% new random seed????
% p(y | x)
Wx = W*x;
p_y_given_x = zeros(4,1);
for y_ = 0:3
  p_y_given_x(y_+1,1) = exp( d(y_+1,1) + sum(log(1 + exp( Wx + c + U(:,y_+1)))));
end
p_y_given_x = p_y_given_x / sum(p_y_given_x);



%% calculate gradients
deriv_output = p_y_given_x - y_vec;
deriv_d = deriv_output;
deriv_W = 0*W;
deriv_U = 0*U;
deriv_c = 0*c;
deriv_b = 0*b; % Discriminative gradient of b is always 0

for y_ = 0:3
  tmp = exp( Wx + c + U(:,y_+1));
  deriv_hid = deriv_output(y_+1)*tmp./(1+tmp);
  deriv_c = deriv_c + deriv_hid;
  deriv_U(:,y_+1) = deriv_hid;
end
deriv_W = deriv_c*x';

% i will use h_pos instead of p_h_given_y_x
W_gen = p_h_given_y_x * x' - p_h_given_y_x_neg * x_neg';
U_gen = p_h_given_y_x * y_vec' - p_h_given_y_x_neg * y_vec_neg';
b_gen = x - x_neg;
c_gen = p_h_given_y_x - p_h_given_y_x_neg;
d_gen = y_vec - y_vec_neg;



%%%% TEST RBMGENERATIVE
debug = 1;
rng('default');rng(1);
[grads_gen,curr_err,chains,chainsy] =  rbmgenerative(rbm,x',opts.y_train,opts,[],[],debug);


% assert intermediates of rbm generative
load('test_rbmgenerative.mat')
delete('test_rbmgenerative.mat')
assert(isequal(x,v0'))            
assert(isequal(p_h_given_y_x,h0'))
assert(isequal(h_pos,h0_rnd'))        
assert(all(abs(p_x_given_h-vkx_sigm') < 10^-15)) 
assert(isequal(x_neg,vkx') )
assert(isequal(p_y_given_h,vky_prob') )
assert(isequal(y_vec_neg,vky') )
assert(all(abs(p_h_given_y_x_neg-hk') < 10^-15))
assert(isequal(h_neg,hk_sample'))

% assert generative grads
assert(isequal(W_gen,grads_gen.dw))
assert(isequal(U_gen,grads_gen.du))
assert(isequal(b_gen,grads_gen.db))
assert(isequal(c_gen,grads_gen.dc))
assert(isequal(d_gen,grads_gen.dd))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  TEST RBMDISCRIMINATIVE  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check intermediate calculations
[grads_dis,curr_err,chains,chainsy] =  rbmdiscriminative(rbm,x',opts.y_train,opts,[],[],debug);
dis = load('test_rbmdiscriminative.mat');
delete('test_rbmdiscriminative.mat');

assert(all(abs(p_y_given_x-dis.p_y_given_x') < 10^-15))
assert(all(abs(deriv_output-(-dis.dd)) < 10^-15))

%% check discrmininative grads
assert(all(abs(-deriv_W(:)-grads_dis.dw(:)) < 10^-14))
assert(all(abs(-deriv_U(:)-grads_dis.du(:)) < 10^-14))
assert(isequal(-deriv_b,grads_dis.db))
assert(all(abs(-deriv_c-grads_dis.dc) < 10^-14))
assert(all(abs(-deriv_d-grads_dis.dd) < 10^-14))




lambda = 0.1; % generative learning weight
lr = 0.1;

W_hybrid = -deriv_W + lambda*W_gen;
U_hybrid = -deriv_U + lambda*U_gen;
b_hybrid = -deriv_b + lambda*b_gen;
c_hybrid = -deriv_c + lambda*c_gen;
d_hybrid = -deriv_d + lambda*d_gen;

rng('default');rng(1);
opts.hybrid_alpha = lambda;   %alpha correspond with  Learning algorithms for the classification restricted boltzmann machine
[grads_hybrid,curr_err,chains,chainsy] =  rbmhybrid(rbm,x',opts.y_train,opts,[],[],debug);
hyp = load('test_rbmhybrid.mat');
delete('test_rbmhybrid.mat');

% check intermediates of hybrid calculation

% discriminiative grads
assert(all(abs(-deriv_W(:)-hyp.grads_dis.dw(:)) < 10^-14))
assert(all(abs(-deriv_U(:)-hyp.grads_dis.du(:)) < 10^-14))
assert(isequal(-deriv_b,hyp.grads_dis.db))
assert(all(abs(-deriv_c-hyp.grads_dis.dc) < 10^-14))
assert(all(abs(-deriv_d-hyp.grads_dis.dd) < 10^-14))

% generative grads
assert(isequal(W_gen,hyp.grads_gen.dw))
assert(isequal(U_gen,hyp.grads_gen.du))
assert(isequal(b_gen,hyp.grads_gen.db))
assert(isequal(c_gen,hyp.grads_gen.dc))
assert(isequal(d_gen,hyp.grads_gen.dd))

% check hybrid grads
assert(all(abs(W_hybrid(:) - grads_hybrid.dw(:))<10^-14))
assert(all(abs(U_hybrid(:) - grads_hybrid.du(:))<10^-14))
assert(all(abs(b_hybrid(:) - grads_hybrid.db(:))<10^-14))
assert(all(abs(c_hybrid(:) - grads_hybrid.dc(:))<10^-14))
assert(all(abs(d_hybrid(:) - grads_hybrid.dd(:))<10^-14))

W = W + lr*(W_hybrid);
U = U + lr*(U_hybrid);

b = b + lr*(b_hybrid);
c = c + lr*(c_hybrid);
d = d + lr*(d_hybrid);



opts.numepochs = 1;
opts.x_val = [];
opts.test_interval = 2;
rbm.early_stopping = 0;
rbm.train_func = @rbmhybrid;
rbm.momentum = @(epoch) 0;
rbm.learningrate =  @(epoch,curMomentum) lr;
rbm.L1 = 0; rbm.L2 = 0; rbm.sparsity = 0; rbm.L2norm = 0;
rbm.vW = zeros(size(rbm.W));
rbm.vU = zeros(size(rbm.U));
rbm.vb = zeros(size(rbm.b));
rbm.vc = zeros(size(rbm.c));
rbm.vd = zeros(size(rbm.d));
rbm.error = [];


% check rbmapply grads
new_rbm = rbmapplygrads(rbm,grads_hybrid,x',opts.y_train,1);

assert(all(abs(W(:) - new_rbm.W(:))<10^-14))
assert(all(abs(U(:) - new_rbm.U(:))<10^-14))
assert(all(abs(b(:) - new_rbm.b(:))<10^-14))
assert(all(abs(c(:) - new_rbm.c(:))<10^-14))
assert(all(abs(d(:) - new_rbm.d(:))<10^-14))




% For semi-supervised learning (i.e. y is unknown), simply replace
% it by a sample from the current estimate of p(y|x)
% 
% y_vec = zeros(4,1);
% p_y_given_x;
% sumsofar = 0;
% r=rand();
% done=0;
% for y_=0:3
%   sumsofar = sumsofar + p_y_given_x(y_+1);
%   if sumsofar >= r & done==0
%     y_vec(y_+1) = 1;
%     done = 1;
%   end
% end

% and just use the same operations as defined above for obtaining
% your CD statistics to do a CD parameter update.