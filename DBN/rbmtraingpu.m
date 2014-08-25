function hrbm = rbmtraingpu(hrbm, hx_train,opts)
%RBMTRAINGPU trains a single RBM on gpu hybrid - fragile implementation
%
% NOTATION:
% data  : all data given as      [n_samples   x #vis]
%    W  : vis - hid weights      [ #hid       x #vis ]
%    U  : label - hid weights    [ #hid       x #n_classes ]
%    b  : bias of visible layer  [ #vis       x 1]
%    c  : bias of hidden layer   [ #hid       x 1]
%    d  : bias of label layer    [ #n_classes x 1]
%
% See also DBNTRAIN
%
% Copyright Søren Sønderby June 2014

% SETUP and checking
assert(isfloat(hx_train), 'x must be a float');
assert(all(hx_train(:)>=0) && all(hx_train(:)<=1), 'all data in x must be in [0:1]');
n_samples = size(hx_train, 1);

hy_train = opts.y_train;
hx_val = opts.x_val;
hy_val = opts.y_val;
hx_semisup = opts.x_semisup;


%% create GPU batches
batch_idx = createminibatches(opts.gpubatch,hx_train);
num_batches = size(batch_idx,1);

% semisup batches
% semisup batches we enforce that all batches are full size
batch_idx_semisup = createminibatches(opts.gpubatch,hx_semisup);
num_batches_semisup = size(batch_idx_semisup,1);


earlystop.best_err = Inf;


herrors = struct();
herrors.train = [];
herrors.val =[];
herrors.reconerror = [];

%% copy rbm to GPU
drbm  = hrbm.cpToGPU( hrbm);
cur_batch_semisup = 0;

dchx = drbm.pcdchainsx;
dchy = drbm.pcdchainsy;
if drbm.beta > 0
    dchx_s = drbm.pcdchainsx_semisup;
    dchy_s = drbm.pcdchainsy_semisup;
end
    


for epoch = 1 : opts.numepochs
    %profile on -detail builtin; profile clear;
    %profile on; profile clear;
    
    epochtimer = tic;
    
    %% update training
    drbm.curcdn = drbm.cdn(epoch);
    
    curr_err = 0;  % reset error every epoch
    for cur_batch = 1:num_batches
        
        %% keep track of batches for x and x_semisup
        d_start = batch_idx(cur_batch,1);
        d_end   = batch_idx(cur_batch,2);
        dx_train = drbm.array( hx_train(d_start:d_end, : ) );
        dy_train = drbm.array( hy_train(d_start:d_end, : ) );
        
        if drbm.beta > 0
            cur_batch_semisup = cur_batch_semisup + 1;
            cur_batch_semisup = mod(cur_batch_semisup-1,num_batches_semisup) +1;
            start_semisup     = batch_idx_semisup(cur_batch_semisup,1);
            end_semisup       = batch_idx_semisup(cur_batch_semisup,2);
            dx_semisup   = drbm.array( hx_semisup(start_semisup:end_semisup, : ) );
        end
        
        
        
        
        %% iter over samples in gpubatch
        for i = 1:(d_end - d_start + 1)
            drbm.i = i;
            
            % online learning, take single sample of y and y. d means device
            dx = dx_train(i,:);
            dy = dy_train(i,:);
            
            
            %% apply dropout and dropconnect
            % copy original weights if dropconnect is enabled
            if drbm.dropconnect > 0
                W_org = drbm.W;   
                U_org = drbm.U; 
                c_org = drbm.c; 
            end
            
            if drbm.dropout > 0
                drbm.dropout_mask =  drbm.rand([1,size(drbm.W,1)]) > drbm.dropout;
            end
            
            if drbm.dropconnect > 0
                drbm.W = drbm.W .* (drbm.rand(size(drbm.W)) > drbm.dropconnect);
                drbm.U = drbm.U .* (drbm.rand(size(drbm.U)) > drbm.dropconnect);
                drbm.c = drbm.c .* (drbm.rand(size(drbm.c)) > drbm.dropconnect);
            end
                
            
            %% calculate weights
            tcwx = bsxfun(@plus,drbm.c',dx * drbm.W');
            
            % generative
            if drbm.alpha > 0
                [dw_g,db_g,dc_g,du_g,dd_g,vkx,dchx,dchy]    = generative(drbm,dx,dy,tcwx,dchx,dchy);
                dw_ = drbm.alpha * dw_g;
                dc_ = drbm.alpha * dc_g;
                du_ = drbm.alpha * du_g;
                dd_ = drbm.alpha * dd_g;
                db_ = drbm.alpha * db_g;
                
                curr_err = curr_err +  sum(sum((dx - vkx) .^ 2));
            end
            
            % discriminative
            if drbm.alpha < 1
                [dw_d,dc_d,du_d,dd_d,p_y_given_x] = discriminative(drbm,dx,dy,tcwx');
                dw_ =(1-drbm.alpha) * dw_d;
                dc_ =(1-drbm.alpha) * dc_d;
                du_ =(1-drbm.alpha) * du_d;
                dd_ =(1-drbm.alpha) * dd_d;
                db_ = drbm.zeros(size(drbm.b));
            end
            
            % semisupervised weights
            if drbm.beta > 0
                i_semisup = mod(i-1,size(dx_semisup,1)) +1; % semisup sample counter
                dx_s = dx_semisup(i_semisup,:);
                if drbm.alpha == 1
                    % if no discrminative we need to calculate p_y_given_x
                    [~,~,~,~,p_y_given_x] = discriminative(drbm,dx,dy,tcwx');
                end
                dy_s = samplevec(drbm, p_y_given_x);  % sample from current y|x
                tcwx_semisup = bsxfun(@plus,drbm.c',dx_s * drbm.W');
                [dw_s,db_s,dc_s,du_s,dd_s,~,dchx_s,dchy_s]    = generative(drbm,dx_s,dy_s,tcwx_semisup,dchx_s,dchy_s);
                dw_ =drbm.beta * dw_s;
                dc_ =drbm.beta * dc_s;
                du_ =drbm.beta * du_s;
                dd_ =drbm.beta * dd_s;
                db_ =drbm.beta * db_s;
            end
            
            %% restore original weighs if dropconnect
            if  drbm.dropconnect > 0
                drbm.W = W_org;   
                drbm.U = U_org; 
                drbm.c = c_org; 
            end
            
            
            %% regularization
            % sparsity
            if drbm.sparsity > 0
                %db_ = bsxfun(@minus,db_,drbm.sparsity);
                dc_ = bsxfun(@minus,dc_,drbm.sparsity);
                %dd_ = bsxfun(@minus,dd_,drbm.sparsity);
            end
            
            % L2
            if drbm.L2 > 0
                dw_ = bsxfun(@minus,dw_,drbm.L2 .* dw_);
                du_ = bsxfun(@minus,du_,drbm.L2 .* du_);
            end
            
            % L1
            if drbm.L1 > 0
                dw_ = bsxfun(@minus,dw_,drbm.L1 .* sign(dw_));
                du_ = bsxfun(@minus,du_,drbm.L1 .* sign(du_));
            end
            
            %%  momentum
            drbm.vW = drbm.momentum(epoch) * drbm.vW + drbm.learningrate(epoch) * dw_;
            drbm.vc = drbm.momentum(epoch) * drbm.vc + drbm.learningrate(epoch) * dc_;
            drbm.vb = drbm.momentum(epoch) * drbm.vb + drbm.learningrate(epoch) * db_;
            drbm.vU = drbm.momentum(epoch) * drbm.vU + drbm.learningrate(epoch) * du_;
            drbm.vd = drbm.momentum(epoch) * drbm.vd + drbm.learningrate(epoch) * dd_;
            
            %%  Gradients
            drbm.W = drbm.W + drbm.vW;
            drbm.b = drbm.b + drbm.vb;
            drbm.c = drbm.c + drbm.vc;
            drbm.U = drbm.U + drbm.vU;
            drbm.d = drbm.d + drbm.vd;
            
            
            
%              if mod(i,100) == 0
%                  hist(drbm.U(:),100)
%                  pause;            
%              end
            
            
            
            
            
            drbm.reconerror  = curr_err;
            
            %% clear data
            %             clear dx dy dx_s                            % sample data
            %             clear dw_ db_ dc_ du_ dd_                        % gradients
            %             clear dw_g db_g dc_g du_g dd_g vkx tcwx     % generativ
            %             clear dw_d dc_d du_d dd_d p_y_given_x       % discrminative
            %             clear dw_s db_s dc_s du_s dd_s tcwx_semisup % semisup
        end
        
        % clear gpu batches
        %         clear dx_train dy_train dx_semisup
    end
    epochtime = toc(epochtimer);
    
    
    if mod(epoch,drbm.testinterval) == 0
        valtimer = tic;
        hrbm = drbm.cpToHOST(drbm);
        
        % update errors on cpu
        [herrors,errstr] =rbmcalculateerrors(hrbm,herrors,epoch,...
            hx_train,hx_val,hy_train,hy_val,epochtime);
        
        %% Earlystopping
        if  earlystop.best_err >= herrors.val(end)
            earlystop.best_str = ' ***';
            earlystop.best_err = herrors.val(end);
            earlystop.best_rbm = hrbm;
            earlystop.best_rbm.herrors = herrors;
            earlystop.patience = hrbm.patience;
            if ~isempty(opts.outfile)
                save(['temp_' opts.outfile],'hrbm','herrors');
            end
        else
            earlystop.best_str = '';
            earlystop.patience = earlystop.patience-opts.testinterval;
        end
        
        %% print output
        if opts.gpu == 1
            gpu_mem  = ['. GPU mem free: ' num2str(opts.thisgpu.FreeMemory / 10^9)];
        else
            gpu_mem = '';
        end
        valtime = toc(valtimer);
        valtime = ['  -   testing time: ' num2str(valtime)];
        disp([errstr,valtime,gpu_mem,  earlystop.best_str]);
        
        %profsave(profile('info'),['prof_' num2str(epoch)]);
        % stop training?
        if earlystop.patience < 0 || epoch == opts.numepochs
            hrbm = earlystop.best_rbm;
            break
        end
    else
        disp(['Epoch ', num2str(epoch) ' - TIME:' num2str(epochtime)]);
    end
    
    %    job = batch(@rbmcalculateerrors,2,{ hrbm,herrors,epoch,...
    %        hx_train,hx_val,hy_train,hy_val,time});
    
end

disp('Training Done')
if ~isempty(opts.outfile)
    save(opts.outfile,'hrbm','herrors');
end

end









