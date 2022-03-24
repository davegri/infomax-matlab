
global DATA_PATH;
global OUTPUT_PATH;
DATA_PATH = "./Data";
OUTPUT_PATH = "./Output";
mkdir(OUTPUT_PATH);

%% Perform ICA on 3 noisless mixes
[mixes, fs] = load_audio_files("mix", 3);
C = infomax(mixes, fs, "unmix");

%% Perform ICA on 3 noisy mixes
[noisy_mixes, nfs] = load_audio_files("noisy_mix", 3);
C_noisy = infomax(noisy_mixes, nfs, "noisy_unmix");

%% perform PCA on 7 noisy mixes
% Load noisy mixes
[noisy_mixes, nfs] = load_audio_files("noisy_mix", 7);

% Sanger PCA
pca_mat_sang = sanger_pca(noisy_mixes - mean(noisy_mixes), 3);

% Analytic PCA
pca_mat = pca(noisy_mixes, 'NumComponents', 3);

% Calculate distance network sanger PCA and analytic PCA
pca_dist=norm(abs(pca_mat_sang)-abs(pca_mat'));

%% perform ICA on 3 denoised mixes
denoised_mix = noisy_mixes*pca_mat_sang';
C_denoised = infomax(denoised_mix, nfs, "denoised_unmixed");

%% Show results
t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
title(t, "abs of correlation between sources and ICA results");

nexttile
heatmap(abs(C));
title("non-noisy mixes");

nexttile
heatmap(abs(C_noisy))
title("noisy mixes");

nexttile
heatmap(abs(C_denoised))
title("mixes denoised using PCA");

% Specify common labels and spacing
xlabel(t, 'ICA result')
ylabel(t, 'source')

% unmix mixed audio files using infomax algorithm
function C = infomax(mixes, fs, unmix_filename)
    global OUTPUT_PATH
    
    % perform ICA using infomax
    ica_mat = getInfomaxMat(mixes);
    unmixed = mixes*ica_mat';
    
    % normalize audio data between -1,1
    norm_unmixed = unmixed./max(abs(unmixed));
    
    % save audio files to disk
    formatSpec = './%s%d.wav';
    for j=1:size(unmixed,2)
        filename = sprintf(formatSpec,unmix_filename, j);
        path = fullfile(OUTPUT_PATH, filename);
        audiowrite(path, norm_unmixed(:,j), fs);
    end
    
    % load ground truth sources, and calculate correlation
    sources = load_audio_files("source", 3);
    C = corr(sources, unmixed);
end

% loads all audio files according to the format [prefix][i].wav for all i
% in 1:num inside the location DATA_PATH
function [data, fs] = load_audio_files(prefix, num)
    global DATA_PATH
    formatSpec = './%s%d.wav';
    for i=1:num
        filename = sprintf(formatSpec, prefix, i);
        path = fullfile(DATA_PATH, filename);
        [data(:,i), fs] = audioread(path);
    end
end


