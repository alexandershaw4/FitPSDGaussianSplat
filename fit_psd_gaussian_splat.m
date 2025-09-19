function Fit = fit_psd_gaussian_splat(f, S, opts)
% FIT_PSD_GAUSSIAN_SPLAT
% Decompose a 1D power spectrum S(f) into K Gaussians + a 1/f background.
%
% Pipeline:
%   1) Initialisation: seed K0 Gaussians (from peaks if possible, else grid)
%   2) Forward "splat": render sum of Gaussians + background to predict S_hat
%   3) Optimisation: Adam on a smooth loss (MSE in linear or log-power domain)
%   4) Pruning: drop components with small contribution, optional re-optimise
%
% Inputs
%   f      [Nx1] frequency axis (Hz), strictly increasing, f>0
%   S      [Nx1] power spectrum (linear power by default)
%   opts   struct with fields (all optional):
%          .use_log_power   (default=false) optimise in log domain
%          .K0              (default=min(50, max(5, floor(numel(f)/10))))
%          .peak_prom       (default=0.02*max(S) or auto if log domain)
%          .max_iter        (default=800)
%          .lr              (default=0.05)    % Adam learning rate
%          .beta1           (default=0.9)     % Adam beta1
%          .beta2           (default=0.999)   % Adam beta2
%          .eps_hat         (default=1e-8)    % Adam epsilon
%          .prune_frac      (default=0.01)    % keep comps >=1% norm share
%          .reopt_after_prune (default=true)  % short re-optimisation
%          .reopt_iter      (default=200)
%          .sigma_init_hz   (default=0.5)     % initial width (Hz)
%          .seed            (default=42)
%
% Outputs (struct Fit)
%   .mu, .sigma, .A      [Kx1] fitted Gaussian centres (Hz), widths (Hz), amplitudes
%   .bg                  struct with fields:
%                        .c (scale >=0), .chi (>=0), .b (offset >=0)
%   .S_hat               [Nx1] final reconstructed spectrum
%   .S_bg                [Nx1] background alone
%   .S_peaks             [NxK] each Gaussian contribution
%   .keep_idx            indices of surviving Gaussians after pruning
%   .history             loss curve
%   .opts                options actually used
%
% Example:
%   Fit = fit_psd_gaussian_splat(f, S, struct('use_log_power',true));
%   plot(f, S, '-', f, Fit.S_hat, '--'); legend raw fitted
%
% Dr Alexander D. Shaw — 1D “Gaussian splatting” for spectra

% ---------- coerce & shape ----------
f = f(:);
S = S(:);

if nargin < 3 || isempty(opts) || ~isstruct(opts)
    opts = struct;
end

% Safe getter that does NOT evaluate both branches
    function val = getopt_local(name, def)
        if isfield(opts, name) && ~isempty(opts.(name))
            val = opts.(name);
        else
            val = def;
        end
    end

% ---------- defaults ----------
use_log_power     = getopt_local('use_log_power', false);
K0                = getopt_local('K0',           min(50, max(5, floor(numel(f)/10))));
max_iter          = getopt_local('max_iter',     800);
lr                = getopt_local('lr',           0.05);
beta1             = getopt_local('beta1',        0.9);
beta2             = getopt_local('beta2',        0.999);
eps_hat           = getopt_local('eps_hat',      1e-8);
prune_frac        = getopt_local('prune_frac',   0.01);
reopt_after_prune = getopt_local('reopt_after_prune', true);
reopt_iter        = getopt_local('reopt_iter',   200);
sigma_init_hz     = getopt_local('sigma_init_hz', 0.5);
seed              = getopt_local('seed',         42);

if use_log_power
    peak_prom = getopt_local('peak_prom', 0.02 * max(log1p(S)));
else
    peak_prom = getopt_local('peak_prom', 0.02 * max(S));
end

% ---------- sanity checks ----------
assert(numel(f)==numel(S), 'f and S must be same length.');
assert(all(diff(f)>0), 'f must be strictly increasing.');
assert(all(f>0), 'f must be > 0 for 1/f background.');

% ---------- target (log or lin) ----------
if use_log_power
    T = log1p(S);                 % log(1+S) for stability
    target = @(X) log1p(max(X, 0)); % forward maps to compare in log domain
else
    T = S;
    target = @(X) X;
end
N = numel(f);
fmin = f(1); fmax = f(end);

% ---------- initial peaks ----------
rng(seed);
% Try findpeaks if available; otherwise grid seeds
Kcand = K0;
mu0 = linspace(fmin, fmax, Kcand)';
A0  = interp1(f, S, mu0, 'linear', 'extrap');
% refine with peaks when possible
try
    if use_log_power
        [pks, locs] = findpeaks(log1p(S), f, 'MinPeakProminence', peak_prom);
    else
        [pks, locs] = findpeaks(S, f, 'MinPeakProminence', peak_prom);
    end
    if ~isempty(locs)
        % Mix detected peaks with grid to reach ~K0
        np = min(numel(locs), K0);
        mu0 = sort([locs(:); linspace(fmin, fmax, K0-np)']);
        mu0 = unique(mu0(:), 'stable');
        if numel(mu0) > K0, mu0 = mu0(1:K0); end
        Acand = interp1(f, S, mu0, 'linear', 'extrap');
        A0 = max(Acand, 0.1 * max(S));
    end
catch
    % ignore if Signal Toolbox not present
end
K = numel(mu0);
sigma0 = sigma_init_hz * ones(K,1);

% ---------- background init: c * f^{-chi} + b ----------
% crude fit: line in log-log for aperiodic; then clamp to >=0
logf = log(f);
logS = log(max(S, eps));
p    = polyfit(logf, logS, 1);   % log S ~ a + b log f
chi0 = max(-p(1), 0);
c0   = max(exp(p(2)), 1e-6);
b0   = max(min(S), 0);

% ---------- reparameterise to unconstrained ----------
% Softplus: sp(x)=log(1+exp(x)) keeps positive values
sp     = @(x) log1p(exp(x));
sp_inv = @(y) log(exp(y)-1+1e-12);   % inverse softplus (approx)
sigmoid = @(x) 1 ./ (1 + exp(-x));

% mu in [fmin,fmax] via sigmoid
mu_raw     = logit((mu0 - fmin)/(fmax - fmin)); % inverse-sigmoid init
sigma_raw  = sp_inv(sigma0);
A_raw      = sp_inv(A0);
c_raw      = sp_inv(c0);
chi_raw    = sp_inv(chi0);
b_raw      = sp_inv(b0);

% Adam buffers
m = zeros(size([mu_raw; sigma_raw; A_raw; c_raw; chi_raw; b_raw]));
v = m;
tstep = 0;

% ---------- main optimisation ----------
history = zeros(max_iter,1);
for it = 1:max_iter
    % forward
    [S_hat, S_bg, S_peaks, params] = forward_render(mu_raw, sigma_raw, A_raw, c_raw, chi_raw, b_raw, f, fmin, fmax, sp, sigmoid);
    R = target(S_hat) - T;
    loss = 0.5 * mean(R.^2);
    history(it) = loss;

    % gradients by hand (efficient & stable)
    %[g_mu, g_sigma, g_A, g_c, g_chi, g_b] = backward_grad(R, f, target, S_hat, S_bg, S_peaks, params, sp, sigmoid);

    [g_mu, g_sigma, g_A, g_c, g_chi, g_b] = backward_grad( ...
    R, f, use_log_power, S_hat, S_bg, S_peaks, params, ...
    mu_raw, sigma_raw, A_raw, c_raw, chi_raw, b_raw, fmin, fmax);

    % pack grads
    g = [g_mu; g_sigma; g_A; g_c; g_chi; g_b];

    % Adam update
    tstep = tstep + 1;
    m = beta1*m + (1-beta1)*g;
    v = beta2*v + (1-beta2)*(g.^2);
    mhat = m ./ (1 - beta1^tstep);
    vhat = v ./ (1 - beta2^tstep);
    step = lr * mhat ./ (sqrt(vhat) + eps_hat);

    % unpack and step
    idx = 0;
    [mu_raw, idx]    = take_step(mu_raw,    step, idx);
    [sigma_raw, idx] = take_step(sigma_raw, step, idx);
    [A_raw, idx]     = take_step(A_raw,     step, idx);
    [c_raw, idx]     = take_step(c_raw,     step, idx);
    [chi_raw, idx]   = take_step(chi_raw,   step, idx);
    [b_raw, ~]       = take_step(b_raw,     step, idx);

    % small early stopping heuristic
    if it > 50 && abs(history(it) - history(it-1)) < 1e-9
        history = history(1:it);
        break;
    end
end

% ---------- pruning ----------
[~, ~, S_peaks, params] = forward_render(mu_raw, sigma_raw, A_raw, c_raw, chi_raw, b_raw, f, fmin, fmax, sp, sigmoid);
S_comb = sum(S_peaks, 2);
peak_norms = sqrt(sum(S_peaks.^2, 1));             % L2 norm of each component
total_norm = norm(S_comb) + eps;
frac = peak_norms / total_norm;
keep_idx = find(frac >= prune_frac);
if isempty(keep_idx) && ~isempty(peak_norms)
    [~, keep_idx] = max(peak_norms); % keep the strongest if all tiny
end

% reduce parameters
mu_raw     = mu_raw(keep_idx);
sigma_raw  = sigma_raw(keep_idx);
A_raw      = A_raw(keep_idx);
K = numel(keep_idx);

% optional short re-optim
if reopt_after_prune && K>0
    m = zeros(size([mu_raw; sigma_raw; A_raw; c_raw; chi_raw; b_raw]));
    v = m; tstep = 0;
    for it = 1:reopt_iter
        [S_hat, S_bg, S_peaks] = forward_render(mu_raw, sigma_raw, A_raw, c_raw, chi_raw, b_raw, f, fmin, fmax, sp, sigmoid);
        R = target(S_hat) - T;
        %[g_mu, g_sigma, g_A, g_c, g_chi, g_b] = backward_grad(R, f, target, S_hat, S_bg, S_peaks, [], sp, sigmoid);
        
        [g_mu, g_sigma, g_A, g_c, g_chi, g_b] = backward_grad( ...
            R, f, use_log_power, S_hat, S_bg, S_peaks, [], ...
            mu_raw, sigma_raw, A_raw, c_raw, chi_raw, b_raw, fmin, fmax);

        g = [g_mu; g_sigma; g_A; g_c; g_chi; g_b];
        tstep = tstep + 1;
        m = beta1*m + (1-beta1)*g;
        v = beta2*v + (1-beta2)*(g.^2);
        mhat = m ./ (1 - beta1^tstep);
        vhat = v ./ (1 - beta2^tstep);
        step = lr * mhat ./ (sqrt(vhat) + eps_hat);
        idx = 0;
        [mu_raw, idx]    = take_step(mu_raw,    step, idx);
        [sigma_raw, idx] = take_step(sigma_raw, step, idx);
        [A_raw, idx]     = take_step(A_raw,     step, idx);
        [c_raw, idx]     = take_step(c_raw,     step, idx);
        [chi_raw, idx]   = take_step(chi_raw,   step, idx);
        [b_raw, ~]       = take_step(b_raw,     step, idx);
    end
end

% ---------- final forward & unpack ----------
[S_hat, S_bg, S_peaks, P] = forward_render(mu_raw, sigma_raw, A_raw, c_raw, chi_raw, b_raw, f, fmin, fmax, sp, sigmoid);

Fit.mu     = P.mu;
Fit.sigma  = P.sigma;
Fit.A      = P.A;
Fit.bg     = struct('c', P.c, 'chi', P.chi, 'b', P.b);
Fit.S_hat  = S_hat;
Fit.S_bg   = S_bg;
Fit.S_peaks= S_peaks;
Fit.keep_idx = keep_idx;
Fit.history  = history;
Fit.opts   = struct('use_log_power',use_log_power,'K_final',K, ...
                    'max_iter',max_iter,'lr',lr,'prune_frac',prune_frac);

% ========= nested helpers =========
function [S_hat, S_bg, S_peaks, P] = forward_render(mu_r, sig_r, A_r, c_r, chi_r, b_r, f, fmin, fmax, sp, sigmoid)
    mu    = fmin + (fmax - fmin) * sigmoid(mu_r);
    sigma = sp(sig_r);
    A     = sp(A_r);
    c     = sp(c_r);
    chi   = sp(chi_r);
    b     = sp(b_r);

    % Gaussians
    F = f(:);
    Kloc = numel(mu);
    S_peaks = zeros(numel(F), Kloc);
    for k = 1:Kloc
        S_peaks(:,k) = A(k) * exp(-0.5*((F - mu(k))./max(sigma(k),1e-9)).^2);
    end
    % Background
    S_bg = c * (F.^(-chi)) + b;

    S_hat = S_bg + sum(S_peaks,2);
    if nargout>3
        P = struct('mu',mu,'sigma',sigma,'A',A,'c',c,'chi',chi,'b',b);
    end
end

function [g_mu_raw, g_sigma_raw, g_A_raw, g_c_raw, g_chi_raw, g_b_raw] = backward_grad( ...
        R, f, use_log_power, S_hat, S_bg, S_peaks, P, ...
        mu_raw, sigma_raw, A_raw, c_raw, chi_raw, b_raw, fmin, fmax)

    % Derivatives for reparameterisations
    sigmoid = @(x) 1 ./ (1 + exp(-x));
    % softplus'(x) = sigmoid(x)
    dsp = @(x) sigmoid(x);

    % dL/dS_hat
    N = numel(R);
    if use_log_power
        % L = 0.5 * mean((log(1+S_hat)-T)^2)
        % dL/dS_hat = (R/N) * d log(1+S)/dS = (R/N) * 1/(1+S_hat)
        dL_dShat = (R ./ max(1 + S_hat, 1e-12)) / N;
    else
        % L = 0.5 * mean((S_hat - T)^2)
        dL_dShat = R / N;
    end

    F = f(:);
    Kloc = size(S_peaks, 2);

    % If P (constrained params) was not provided, approximate from raw
    if isempty(P)
        mu    = fmin + (fmax - fmin) * sigmoid(mu_raw);
        sigma = dsp(sigma_raw);              % = softplus(sigma_raw)'
        A     = dsp(A_raw);
        c     = dsp(c_raw);
        chi   = dsp(chi_raw);
        b     = dsp(b_raw);
    else
        mu = P.mu; sigma = P.sigma; A = P.A; c = P.c; chi = P.chi; b = P.b;
    end

    % ----- Gradients w.r.t. constrained Gaussians -----
    g_mu_c    = zeros(Kloc,1);
    g_sigma_c = zeros(Kloc,1);
    g_A_c     = zeros(Kloc,1);

    for k = 1:Kloc
        Gk = S_peaks(:,k);    % = A(k) * exp(-0.5 * ((F-mu)/sigma)^2)
        if isempty(Gk), continue; end

        % Safe helpers
        s  = max(sigma(k), 1e-12);
        Ak = max(A(k),     1e-12);

        % dG/dA = exp(...)  -> equivalently Gk / A
        dG_dA     = Gk / Ak;

        % dG/dmu = G * ((F - mu)/sigma^2)
        dG_dmu    = Gk .* ((F - mu(k)) / (s*s));

        % dG/dsigma = G * ((F - mu)^2 / sigma^3)
        dG_dsigma = Gk .* (((F - mu(k)).^2) / (s^3));

        g_A_c(k)     = sum(dL_dShat .* dG_dA);
        g_mu_c(k)    = sum(dL_dShat .* dG_dmu);
        g_sigma_c(k) = sum(dL_dShat .* dG_dsigma);
    end

    % ----- Background grads (constrained) -----
    % S_bg = c * f^{-chi} + b
    fchi = F.^(-max(chi,0));              % safe power
    dBg_dc   = fchi;
    dBg_dchi = -c * fchi .* log(F);
    dBg_db   = ones(size(F));

    g_c_c   = sum(dL_dShat .* dBg_dc);
    g_chi_c = sum(dL_dShat .* dBg_dchi);
    g_b_c   = sum(dL_dShat .* dBg_db);

    % ----- Chain rule to RAW (unconstrained) params -----
    % mu = fmin + (fmax-fmin) * sigmoid(mu_raw)
    s_mu = sigmoid(mu_raw);
    dmu_dmu_raw = (fmax - fmin) * s_mu .* (1 - s_mu);

    % sigma = softplus(sigma_raw)  -> d sigma / d sigma_raw = sigmoid(sigma_raw)
    dsigma_dsigma_raw = dsp(sigma_raw);

    % A = softplus(A_raw)
    dA_dA_raw = dsp(A_raw);

    % c, chi, b via softplus
    dc_dc_raw   = dsp(c_raw);
    dchi_dchi_raw = dsp(chi_raw);
    db_db_raw   = dsp(b_raw);

    % Apply chain rule
    g_mu_raw    = g_mu_c    .* dmu_dmu_raw;
    g_sigma_raw = g_sigma_c .* dsigma_dsigma_raw;
    g_A_raw     = g_A_c     .* dA_dA_raw;

    g_c_raw   = g_c_c   * dc_dc_raw;
    g_chi_raw = g_chi_c * dchi_dchi_raw;
    g_b_raw   = g_b_c   * db_db_raw;
end


function [xnew, idx] = take_step(x, step, idx)
    xnew = x - step(idx + (1:numel(x)));
    idx  = idx + numel(x);
end

% --- utilities ---
function y = logit(p)
    p = min(max(p, 1e-6), 1-1e-6);
    y = log(p./(1-p));
end

end
