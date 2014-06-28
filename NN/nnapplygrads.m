function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
%
% Modified by Søren Sønderby June 2014

for i = 1 : (nn.n - 1)
    if(nn.weightPenaltyL2>0)
        dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
    else
        dW = nn.dW{i};
    end
    
    dW = nn.currentLearningRate * dW;
    
    if(nn.currentMomentum>0)
        nn.vW{i} = nn.currentMomentum*nn.vW{i} + dW;
        dW = nn.vW{i};
    end
    
    nn.W{i} = nn.W{i} - dW;
    
    
    if nn.weightMaxL2norm > 0;
        L2 = nn.weightMaxL2norm;
        %neruon inputs
        z = sum(nn.W{i}.^2,2);
        %normalization factor
        norm_factor = sqrt(z/L2);
        idx = norm_factor < 1;
        norm_factor(idx) = 1;
        %rescale weights and biases
        nn.W{i} = bsxfun(@rdivide,nn.W{i},norm_factor);
    end
    
    
end
end
