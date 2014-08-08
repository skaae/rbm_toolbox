function dbn_cRBM_tests_gpu()
rng('default');rng(1);

%% init test weights '
x_size = 3;
n_hid = 7;
lr = 0.1;

x = rand(x_size,1);
y = 2;
y_vec = [0 0 1 0]';
W = rand(n_hid,x_size)-0.5;
W = W ./ max(size(W));
U = rand(n_hid,4)-0.5;
U = U ./ max(size(U));
d = zeros(4,1);
c = zeros(n_hid,1);
b = zeros(x_size,1);

 [ opts ] = dbncreateopts();
 opts.y_train = y_vec';
 opts.gpu = -1;  % testing
 opts.classRBM = 1;
dbn = dbnsetup([20], x', opts);

rbm = dbn.rbm{1};

% init RBM with same weights
rbm.classRBM = 1;
rbm.dropouthidden = 0;
rbm.curcdn = 1;
rbm.curMomentum = 0;
rbm.curLR = lr;
rbm.W = W;
rbm.U = U;
rbm.d = d;
rbm.c = c;
rbm.b = b;

% p(h | y, x)  UP
p_h_given_y_x  = 1 ./ (1 + exp(-(W*x + c + U*y_vec)));
h_pos = rbm.rand([n_hid,1]) < p_h_given_y_x;

% p(y | h)  % Vky
p_y_given_h_non_norm= exp( U'*h_pos + d );
p_y_given_h = p_y_given_h_non_norm / sum(p_y_given_h_non_norm);

% p(x | h)
p_x_given_h = 1 ./ (1 + exp(-(W'*h_pos + b)));
x_neg = rbm.rand([x_size,1]) < p_x_given_h;

%% Sample from p(y | h)
y_sample = zeros(4,1);
sumsofar = 0;
r=rbm.rand([1,1]);
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
h_neg = rbm.rand([n_hid,1]) < p_h_given_y_x_neg;


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
tcwx = bsxfun(@plus,rbm.c',x' * rbm.W');
 [dw_g,db_g,dc_g,du_g,dd_g,vkx] =  generative(rbm,x',opts.y_train,tcwx);

chk = @(a,b) all( abs(a(:) - b(:) ) < 10^-12 );
% assert generative grads
assert(chk(W_gen,dw_g))
assert(chk(U_gen,du_g))
assert(chk(b_gen,db_g))
assert(chk(c_gen,dc_g))
assert(chk(d_gen,dd_g))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  TEST RBMDISCRIMINATIVE  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check intermediate calculations
[dw_d,dc_d,du_d,dd_d]  = discriminative(rbm,x',opts.y_train,tcwx');

assert(chk(-deriv_W,dw_d))
assert(chk(-deriv_U,du_d))
assert(chk(-deriv_c,dc_d))
assert(chk(-deriv_d,dd_d))

% dis_test = @() discriminative(rbm,x',opts.y_train,tcwx');  
% gen_test = @() generative(rbm,x',opts.y_train,tcwx);
% 
% timeit(dis_test,4)
% 
% timeit(gen_test,6)

% profile on; profile clear;
% for i = 1:10
% gen_test();
% dis_test();
% end
% profile viewer

disp('dbn_cRB_test_gpu: All test passed')
end


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

