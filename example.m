%% DEMO: Mock M/EEG-like PSD + Gaussian "splat" fit
% Requires: fit_psd_gaussian_splat.m in your MATLAB path.

clear; clc; close all;

% ----- frequency grid -----
w = (1 : 1/4 : 90)';   % 1 Hz to 90 Hz in 0.25-Hz steps
N = numel(w);

% ----- make a mock PSD: 1/f^chi + oscillatory Gaussians + small noise -----
rng(7);

% aperiodic background
chi = 1.2;         % 1/f slope
c   = 5;           % scale
b   = 0.1;         % white floor
bg  = c*(w.^(-chi)) + b;

% oscillatory peaks (theta, alpha, beta, low-gamma)
peaks = [ ...
    6,   0.6, 1.6;   % [mu, sigma, A]
    10,  1.5, 3.5;
    20,  2.0, 1.8;
    40,  3.0, 1.2
];
S_osc = zeros(N,1);
for k = 1:size(peaks,1)
    mu    = peaks(k,1);
    sigma = peaks(k,2);
    A     = peaks(k,3);
    S_osc = S_osc + A * exp(-0.5*((w - mu)/sigma).^2);
end

% add a bit of mains line noise (50 Hz) with a narrow peak (optional)
S_osc = S_osc + 0.8 * exp(-0.5*((w - 50)/0.4).^2);

% small jaggedness to resemble empirical PSDs
jitter = 0.05 * bg .* (0.5 - rand(N,1));

% final mock PSD (ensure non-negative)
psd = max(bg + S_osc + jitter, 0);

% ----- fit with Gaussian-splat routine -----
opts = struct('use_log_power', true, 'K0', 30, 'sigma_init_hz', 0.8, ...
              'max_iter', 1000, 'prune_frac', 0.01, 'reopt_after_prune', true);
Fit = fit_psd_gaussian_splat(w, psd, opts);

% ----- reconstruct -----
[S_hat, ~, S_peaks] = reconstruct_psd(Fit, w);

% ----- plot exactly as requested -----
figure('Color','w'); hold on;
plot(w, psd, 'k--', 'LineWidth', 3);
plot(w, S_hat, 'r*', 'LineWidth', 2.5);
plot(w, S_peaks, 'Color', [0.4 0.4 0.4], 'LineWidth', 2);
xlim([w(1) w(end)]);
xlabel('Frequency (Hz)'); ylabel('Power');
legend('Mock PSD','Fit (Gaussians + 1/f)','Individual Gaussians');
title('Mock M/EEG PSD fit with Gaussian “splatting”');

%%Optional: log-scale view (clearer)
% figure('Color','w'); hold on;
% plot(w, log1p(psd), 'k', 'LineWidth', 3);
% plot(w, log1p(S_hat), 'r', 'LineWidth', 2.5);
% plot(w, log1p(S_peaks), 'Color', [0.4 0.4 0.4], 'LineWidth', 2);
% xlim([w(1) w(end)]); xlabel('Hz'); ylabel('log(1+Power)'); grid on;
% title('Log-domain view');

