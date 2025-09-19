function [S_hat, S_bg, S_peaks] = reconstruct_psd(Fit, f)
% Rebuild PSD from fitted parameters onto arbitrary frequency vector f.
f = f(:);
c   = Fit.bg.c;
chi = Fit.bg.chi;
b   = Fit.bg.b;

K = numel(Fit.mu);
S_peaks = zeros(numel(f), K);
for k = 1:K
    mu    = Fit.mu(k);
    sigma = max(Fit.sigma(k), 1e-12);
    A     = Fit.A(k);
    S_peaks(:,k) = A * exp(-0.5 * ((f - mu)/sigma).^2);
end

S_bg  = c * (f .^ (-chi)) + b;
S_hat = S_bg + sum(S_peaks, 2);
end
