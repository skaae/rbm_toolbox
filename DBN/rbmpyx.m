function [class_prob,F]= rbmpyx(rbm,x,train_or_test)
%RBMPYX calculates class probabilities, helper function
n_classes = size(rbm.d,1);
n_samples = size(x,1);

cwx = bsxfun(@plus,rbm.W*x',rbm.c);

% only check for dropout in training mode
if isequal(lower(train_or_test),'train')
    if rbm.dropout_hidden > 0
        cwx = cwx .* rbm.hidden_mask;
    end
end

F = bsxfun(@plus, permute(rbm.U, [1 3 2]), cwx);
class_log_prob = zeros(n_samples,n_classes);
for y = 1:n_classes
    class_log_prob(:,y) =  sum( softplus(F(:,:,y)), 1)+ rbm.d(y);
end

%normalize probabilities in numerically stable way
class_prob = exp(bsxfun(@minus, class_log_prob, max(class_log_prob, [], 2)));
class_prob = bsxfun(@rdivide, class_prob, sum(class_prob, 2));

end