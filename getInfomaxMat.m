function [W] = getInfomaxMat(M)
    n_samples = size(M,1);
    sample_dim = size(M,2);
    
    % Init ICA network
    in_dim = sample_dim;
    out_dim = in_dim;
    W = randn(out_dim, in_dim);
    sig = @(x) 1./(1 + exp(-x));
    
    % Train ICA network
    eta = 1e-1;
    epochs = 25;
    alpha = 0.98; % momentum parameter    
    batch_size = 50; % batch size, must be divisible by 5.
    mom_grad = 0; % initalize running momentum gradient
    
    for ep = 1:epochs
        rnd_samp_idx = randperm(n_samples);
        for b=1:n_samples/batch_size
     
            % get samples for current batch
            batch_start = batch_size*(b-1) + 1;
            batch_end = batch_size*b;
            idxs = rnd_samp_idx(batch_start:batch_end);
            X = M(idxs,:);
            
            % calculate batch gradient according to ICA update rule
            Y = 1-2*sig(W*X');
            batch_grad = inv(W')+Y*X/batch_size;
            
            % momentum
            mom_grad = batch_grad + alpha * mom_grad; 
            
            % update weights
            deltaW = eta*mom_grad;
            W = W + deltaW;
        end
    end
end

