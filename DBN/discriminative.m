    function [dw,dc,du,dd,p_y_given_x] = discriminative(rbm,dx,dy,cwx)
         %   rbm.dropout_mask = [1 0 0 1 1 0 0];  rbm.dropout = 0.5;% added
        [n_hidden, n_classes] = size(rbm.U);
        F   = bsxfun(@plus, rbm.U, cwx);       % [hid x classes]        
        if rbm.dropout > 0
            F = bsxfun(@times,F,rbm.dropout_mask');  % added
        end

        F_soft = arrayfun(@softplus,F);
        if rbm.dropout > 0
            F_soft = bsxfun(@times,F_soft,rbm.dropout_mask');  % added
        end
        
        p_y_given_x_log_prob = rbm.ones([1 ,n_hidden]) *  F_soft + rbm.d';
        p_y_given_x = exp(bsxfun(@minus, p_y_given_x_log_prob, max(p_y_given_x_log_prob, [], 2)));
        p_y_given_x = bsxfun(@rdivide, p_y_given_x, p_y_given_x * rbm.ones([n_classes,1]));
        
        F_sigm = arrayfun(@sigm, F);
        if rbm.dropout > 0
            F_sigm = bsxfun(@times,F_sigm,rbm.dropout_mask');  % added
        end
        
        
        F_sigm_prob  = bsxfun(@times, F_sigm,p_y_given_x);
        F_sigm_dy = F_sigm * dy';
        F_sigm_prob_ones = F_sigm_prob*rbm.ones(n_classes,1);
        
        % calc gradients
        dw = F_sigm_dy * dx - F_sigm_prob_ones * dx;
        du = - F_sigm_prob + bsxfun(@times,F_sigm, rbm.ones([size(F_sigm_prob,1),1]) *dy );
        dc = -F_sigm_prob_ones +  F_sigm_dy;
        dd = (dy - p_y_given_x)';   
    end