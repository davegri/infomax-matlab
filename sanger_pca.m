function [W] = sanger_pca(M, out_dim)
    n_samples = size(M,1);
    % Init network
    in_dim = size(M,2);
    W = randn(out_dim, in_dim);

    % Train network
    eta = 1e-2;
    epochs = 10;
    alpha = 0.90; % momentum parameter    
    batch_size = 50;
    mom_grad = 0;

    for ep = 1:epochs
        rnd_samp_idx = randperm(size(M,1));
        for b=1:n_samples/batch_size
            % get samples for current batch
            batch_start = batch_size*(b-1) + 1;
            batch_end = batch_size*b;
            idxs = rnd_samp_idx(batch_start:batch_end);
            X = M(idxs,:);
            
            % feed forward
            Y = W*X';

            % update weights according to Sanger's learning rule
            batch_grad = -(Y*X - tril(Y*Y')*W)/batch_size;
            mom_grad = batch_grad + alpha * mom_grad; % momentum
            dW = -eta*mom_grad;
            W = W + dW;
        end
    end
end

