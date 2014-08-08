    function [dw,dc,du,dd,p_y_given_x] = discriminative(rbm,dx,dy,cwx)
        [n_hidden, n_classes] = size(rbm.U);  
        F   = bsxfun(@plus, rbm.U, cwx);
        %p_y_given_x_log_prob = sum(  softplus(F), 1)+rbm.d';
        F_soft = arrayfun(@softplus,F);
        p_y_given_x_log_prob = rbm.ones([1 ,n_hidden]) *  F_soft + rbm.d';
        p_y_given_x = exp(bsxfun(@minus, p_y_given_x_log_prob, max(p_y_given_x_log_prob, [], 2)));
        %p_y_given_x = bsxfun(@rdivide, p_y_given_x, sum(p_y_given_x, 2));
        p_y_given_x = bsxfun(@rdivide, p_y_given_x, p_y_given_x * rbm.ones([n_classes,1]));
        
        F_sigm = arrayfun(@sigm, F);
        F_sigm_prob  = bsxfun(@times, F_sigm,p_y_given_x);
        
        
        F_sigm_dy = F_sigm * dy';
        F_sigm_prob_ones = F_sigm_prob*rbm.ones(n_classes,1);
        
        % calc gradients
        dw = F_sigm_dy * dx - F_sigm_prob_ones * dx;
        du = - F_sigm_prob + bsxfun(@times,F_sigm, rbm.ones([size(F_sigm_prob,1),1]) *dy );
        dc = -F_sigm_prob_ones +  F_sigm_dy;
        dd = (dy - p_y_given_x)';   
    end