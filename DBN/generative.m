function [dw,db,dc,du,dd,vkx,chx,chy]= generative(rbm,dx,dy,tcwx,chx,chy)
% generative weights



%% UP
h0 = arrayfun(@sigm,tcwx +  dy * rbm.U');
h0_rnd = double(h0 > rbm.rand(size(h0)));
%h0_rnd = rbm.array( double(bsxfun(@gt, h0, rbm.rand(size(h0)))));      % sample h0

if rbm.traintype  
    % PCD - Use chains as starting point for gibbs sampling
    ch_idx = rbm.randi(size(chx,1),1);   % pick a chain at random
    tcwx_pcd = bsxfun(@plus,rbm.c',chx(ch_idx,:) * rbm.W');
    hid = arrayfun(@sigm,tcwx_pcd +  chy(ch_idx,:) * rbm.U');
    hid = double(hid > rbm.rand(size(hid)));    
else
    % CD
    hid = h0_rnd;
end


if rbm.dropout > 0
    hid = hid .* rbm.dropout_mask; %% apply dropout
end


for n = rbm.colon(1, (rbm.curcdn - 1) )  % matlab crashes gpu crash with a:b notation
    % go down and up + sample
    visx = arrayfun(@sigm, bsxfun(@plus,rbm.b',hid * rbm.W));
    visx = double(visx > rbm.rand(size(visx))); 
    visy = exp(bsxfun(@plus,rbm.d',hid * rbm.U));
    visy = bsxfun(@rdivide, visy, visy * rbm.ones([size(visy,2),1]));
    visy = samplevec(rbm,visy);
    hid = arrayfun(@sigm, bsxfun(@plus,rbm.c',visx * rbm.W') +  visy*rbm.U');
    hid = double(hid > rbm.rand(size(hid)));
    if rbm.dropout > 0
        hid = hid .* rbm.dropout_mask; %% apply dropout
    end
end
% DOWN
%vkx = sigmrnd(rbm, bsxfun(@plus,rbm.b',hid * rbm.W));
vkx = arrayfun(@sigm,bsxfun(@plus,rbm.b',hid * rbm.W));
vkx = double(vkx > rbm.rand(size(vkx)));


vky = exp(bsxfun(@plus,rbm.d',hid * rbm.U));
vky = bsxfun(@rdivide, vky, vky * rbm.ones([size(vky,2),1]));
vky = samplevec(rbm,vky); % sample visible state
%% up
%hk = sigm(bsxfun(@plus,rbm.c',vkx * rbm.W') +  vky * rbm.U');
hk = arrayfun(@sigm, bsxfun(@plus,rbm.c',vkx * rbm.W') +  vky * rbm.U');%%
if rbm.dropout > 0
    hk = hk .* rbm.dropout_mask; %% apply dropout
end

% update chosen chain if PCD
if rbm.traintype
    chx(ch_idx,:) = vkx;
    chy(ch_idx,:) = vky;
end

%% calculate gradients
dw = h0' * dx - hk' * vkx;
db = (dx - vkx)';
dc = (h0 - hk)';
du = h0' * dy - hk' * vky;
dd = (dy - vky)';
end