function [Sigma_new] = add_task(Sigma, W, opts)
%{

Ref
Yu Zhang & Dit-Yan Yeung
A Convex Formulation for Learning Task Relationships in Multi-Task Learning
%}
    WW = W'*W;
    sizeWW = size(WW);
    col_WW = sizeWW(2);
    
    flat_Sigma = reshape(Sigma, [], 1);
    
    size_Sigma= size(Sigma);
    col = size_Sigma(2);
    col_new = size_Sigma(2)+1;
    
    cvxp = cvx_precision( opts.precision );
    cvx_begin sdp quiet
        variable OmegaNew(col_new, col_new) semidefinite;
        variable OmegaOldvec(col*col, 1);
        variable x(col_new*col_new+1, 1); %vec(OmegaNew, t)
        variable sigma;

        C = vertcat(zeros(col_new*col_new,1),-1);
        % minimize -t
        minimize( C' * x )
        
        subject to
            OmegaNew == reshape(x(1:end-1), col_new, col_new);
            sigma == OmegaNew(end, end);
            OmegaOldvec == reshape(OmegaNew(1:end-1, 1:end-1), col*col, 1);
            OmegaOldvec == (1-sigma).*flat_Sigma;
            (OmegaNew - x(end).*WW) <In> semidefinite(col_WW);
    cvx_end
    cvx_precision( cvxp );
    
    Sigma_new = reshape(x(1:end-1), col_WW, col_WW);
end
