%% Complete 10K RF Dataset Generator for ResNet-18

clear; close all; clc;

%% Configuration
config.sampleRate = 20e6;
config.duration = 1e-3;
config.totalSamples = 10000;
config.batchSize = 150;

config.dataDirectory = 'C:\Users\derri\OneDrive\Dokumente\MATLAB\RF_10K_Optimized_Dataset';

config.signalTypes = {'AM', 'FM', 'Bluetooth', 'WiFi', 'Zigbee', 'Radar', 'LTE', 'Noise'};
config.environments = {'indoor_office', 'indoor_industrial', 'outdoor_urban', 'vehicle_mobile', 'dense_network'};

config.scenarioDistribution = struct(...
    'single_signal', 0.35, ...
    'two_signals', 0.35, ...
    'three_signals', 0.20, ...
    'four_plus_signals', 0.10);

config.wifiBoostFactor = 2.0;
config.bluetoothBoostFactor = 1.5;
config.noiseBoostFactor = 1.5;

fprintf('========================================================\n');
fprintf('  OPTIMIZED 10K DATASET FOR RESNET-18\n');
fprintf('========================================================\n');
fprintf('Total samples: %d\n', config.totalSamples);
fprintf('Estimated time: 7-10 minutes\n\n');

if ~exist(config.dataDirectory, 'dir')
    mkdir(config.dataDirectory);
end

%% Environment Parameters
config.envParams = containers.Map();
config.envParams('indoor_office') = struct('snr', 25, 'iqImbalance', 0.05, 'dcOffset', 0.02, 'phaseNoise', -80, 'cfoError', 10e3);
config.envParams('indoor_industrial') = struct('snr', 18, 'iqImbalance', 0.08, 'dcOffset', 0.04, 'phaseNoise', -75, 'cfoError', 15e3);
config.envParams('outdoor_urban') = struct('snr', 20, 'iqImbalance', 0.06, 'dcOffset', 0.03, 'phaseNoise', -78, 'cfoError', 20e3);
config.envParams('vehicle_mobile') = struct('snr', 15, 'iqImbalance', 0.10, 'dcOffset', 0.05, 'phaseNoise', -70, 'cfoError', 25e3);
config.envParams('dense_network') = struct('snr', 12, 'iqImbalance', 0.07, 'dcOffset', 0.04, 'phaseNoise', -72, 'cfoError', 30e3);

%% Generation
fprintf('Starting generation...\n\n');

tic;
batchCounter = 1;
sampleIdx = 1;

nSingle = round(config.totalSamples * config.scenarioDistribution.single_signal);
nTwo = round(config.totalSamples * config.scenarioDistribution.two_signals);
nThree = round(config.totalSamples * config.scenarioDistribution.three_signals);
nFourPlus = config.totalSamples - (nSingle + nTwo + nThree);

batchData = [];
batchLabels = [];
batchEnvs = {};

wifiStats = struct('total', 0, 'withWiFi', 0);
bluetoothStats = struct('total', 0, 'withBluetooth', 0);
noiseStats = struct('total', 0, 'withNoise', 0);

scenarioTypes = {'single', 'two', 'three', 'four_plus'};
scenarioCounts = [nSingle, nTwo, nThree, nFourPlus];

for scenarioIdx = 1:length(scenarioTypes)
    scenarioType = scenarioTypes{scenarioIdx};
    numSamples = scenarioCounts(scenarioIdx);
    
    fprintf('=== %s signals: %d samples ===\n', upper(scenarioType), numSamples);
    
    for i = 1:numSamples
        envIdx = randi(length(config.environments));
        envName = config.environments{envIdx};
        envParams = config.envParams(envName);
        
        switch scenarioType
            case 'single'
                numSignals = 1;
            case 'two'
                numSignals = 2;
            case 'three'
                numSignals = 3;
            case 'four_plus'
                numSignals = randi([4, 6]);
        end
        
        selectedSignals = selectSignalsWithBoosts(config, numSignals, scenarioType);
        
        wifiStats.total = wifiStats.total + 1;
        if any(strcmp(selectedSignals, 'WiFi'))
            wifiStats.withWiFi = wifiStats.withWiFi + 1;
        end
        
        bluetoothStats.total = bluetoothStats.total + 1;
        if any(strcmp(selectedSignals, 'Bluetooth'))
            bluetoothStats.withBluetooth = bluetoothStats.withBluetooth + 1;
        end
        
        noiseStats.total = noiseStats.total + 1;
        if any(strcmp(selectedSignals, 'Noise'))
            noiseStats.withNoise = noiseStats.withNoise + 1;
        end
        
        mixedSignal = generateMixedSignal(selectedSignals, envParams, config);
        mixedSignal = applyImpairments(mixedSignal, envParams, config);
        
        [s, ~, ~] = spectrogram(mixedSignal, hamming(256), 200, 256, config.sampleRate);
        specData = abs(s);
        
        targetRows = 257;
        targetCols = 39;
        if size(specData, 1) ~= targetRows || size(specData, 2) ~= targetCols
            specData = imresize(specData, [targetRows, targetCols]);
        end
        
        if max(specData(:)) > 0
            specData = specData / max(specData(:));
        end
        
        labelVector = zeros(1, length(config.signalTypes));
        for j = 1:length(selectedSignals)
            idx = find(strcmp(config.signalTypes, selectedSignals{j}));
            if ~isempty(idx)
                labelVector(idx) = 1;
            end
        end
        
        if isempty(batchData)
            batchData = zeros(targetRows, targetCols, config.batchSize);
            batchLabels = zeros(config.batchSize, length(config.signalTypes));
            batchEnvs = cell(config.batchSize, 1);
        end
        
        batchIdx = mod(sampleIdx - 1, config.batchSize) + 1;
        batchData(:, :, batchIdx) = specData;
        batchLabels(batchIdx, :) = labelVector;
        batchEnvs{batchIdx} = envName;
        
        if batchIdx == config.batchSize || sampleIdx == config.totalSamples
            actualSize = batchIdx;
            saveBatchData = batchData(:, :, 1:actualSize);
            saveBatchLabels = batchLabels(1:actualSize, :);
            saveBatchEnvs = batchEnvs(1:actualSize);
            
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            filename = sprintf('rf_10k_batch_%03d_%s.mat', batchCounter, timestamp);
            filepath = fullfile(config.dataDirectory, filename);
            
            signalTypesList = config.signalTypes;
            save(filepath, 'saveBatchData', 'saveBatchLabels', 'saveBatchEnvs', 'signalTypesList', '-v7.3');
            
            fprintf('  Batch %d saved\n', batchCounter);
            
            batchCounter = batchCounter + 1;
            batchData = [];
            batchLabels = [];
            batchEnvs = {};
        end
        
        sampleIdx = sampleIdx + 1;
        
        if mod(i, 500) == 0
            fprintf('  Progress: %d/%d\n', i, numSamples);
        end
    end
end

%% Save Metadata
metadata.totalSamples = config.totalSamples;
metadata.signalTypes = config.signalTypes;
metadata.environments = config.environments;
metadata.scenarioDistribution = config.scenarioDistribution;
metadata.sampleRate = config.sampleRate;
metadata.duration = config.duration;
metadata.spectrogramSize = [257, 39];
metadata.numBatches = batchCounter - 1;
metadata.generationDate = datestr(now);
metadata.wifiPresenceRate = wifiStats.withWiFi / wifiStats.total;
metadata.bluetoothPresenceRate = bluetoothStats.withBluetooth / bluetoothStats.total;
metadata.noisePresenceRate = noiseStats.withNoise / noiseStats.total;

metadataFile = fullfile(config.dataDirectory, 'dataset_metadata.mat');
save(metadataFile, 'metadata');

totalTime = toc;
fprintf('\n========== Complete ==========\n');
fprintf('Time: %.2f minutes\n', totalTime/60);
fprintf('Batches: %d\n', metadata.numBatches);
fprintf('WiFi: %.1f%%\n', 100*metadata.wifiPresenceRate);
fprintf('Bluetooth: %.1f%%\n', 100*metadata.bluetoothPresenceRate);
fprintf('Noise: %.1f%%\n', 100*metadata.noisePresenceRate);
fprintf('==============================\n');

%% ============= HELPER FUNCTIONS =============

function selectedSignals = selectSignalsWithBoosts(config, numSignals, scenarioType)
    signalTypes = config.signalTypes;
    weightedPool = signalTypes;
    
    % WiFi boost
    wifiIdx = find(strcmp(signalTypes, 'WiFi'));
    if ~isempty(wifiIdx)
        extra = repmat({'WiFi'}, 1, round(config.wifiBoostFactor) - 1);
        weightedPool = [weightedPool, extra];
    end
    
    % Bluetooth boost
    btIdx = find(strcmp(signalTypes, 'Bluetooth'));
    if ~isempty(btIdx)
        extra = repmat({'Bluetooth'}, 1, round(config.bluetoothBoostFactor) - 1);
        weightedPool = [weightedPool, extra];
    end
    
    % Noise boost
    noiseIdx = find(strcmp(signalTypes, 'Noise'));
    if ~isempty(noiseIdx)
        extra = repmat({'Noise'}, 1, round(config.noiseBoostFactor) - 1);
        weightedPool = [weightedPool, extra];
    end
    
    if strcmp(scenarioType, 'single')
        r = rand;
        if r < 0.30
            selectedSignals = {'WiFi'};
        elseif r < 0.50
            selectedSignals = {'Bluetooth'};
        elseif r < 0.70
            selectedSignals = {'Noise'};
        else
            nonPriority = signalTypes(~ismember(signalTypes, {'WiFi', 'Bluetooth', 'Noise'}));
            selectedSignals = {nonPriority{randi(length(nonPriority))}};
        end
    else
        selectedSignals = {};
        availableSignals = weightedPool;
        
        for i = 1:numSignals
            if isempty(availableSignals), break; end
            idx = randi(length(availableSignals));
            signal = availableSignals{idx};
            if ~ismember(signal, selectedSignals)
                selectedSignals{end+1} = signal; %#ok<AGROW>
            end
            mask = ~strcmp(availableSignals, signal);
            availableSignals = availableSignals(mask);
        end
    end
end

function mixedSignal = generateMixedSignal(selectedSignals, envParams, config)
    t = (0:config.sampleRate*config.duration-1) / config.sampleRate;
    mixedSignal = zeros(size(t));
    numSignals = length(selectedSignals);
    basePower = 1.0 / sqrt(numSignals);
    
    for i = 1:numSignals
        signalType = selectedSignals{i};
        powerVar = 10^((rand*6 - 3)/20);
        signal = generateSingleSignal(signalType, t, config) * basePower * powerVar;
        mixedSignal = mixedSignal + signal;
    end
    
    mixedSignal = awgn(mixedSignal, envParams.snr, 'measured');
end

function signal = generateSingleSignal(signalType, t, config)
    switch signalType
        case 'AM'
            fc = 1e6; fm = 10e3; m = 0.5;
            signal = (1 + m*cos(2*pi*fm*t)) .* cos(2*pi*fc*t);
            
        case 'FM'
            fc = 5e6; fm = 15e3; kf = 75e3;
            signal = cos(2*pi*fc*t + (kf/fm)*sin(2*pi*fm*t));
            
        case 'Bluetooth'
            fc = 2.44e9; symbolRate = 1e6;
            bits = randi([0 1], 1, ceil(length(t)*symbolRate/config.sampleRate));
            signal = pskmod(bits, 2, pi/4);
            signal = resample(signal, length(t), length(bits));
            
        case 'WiFi'
            fc = 2.45e9; nSubcarriers = 52; nSymbols = 10;
            data = randi([0 3], nSubcarriers, nSymbols);
            ofdmSig = qammod(data(:), 16);
            ofdmSig = reshape(ofdmSig, nSubcarriers, nSymbols);
            signal = ifft(ofdmSig, nSubcarriers);
            signal = signal(:)';
            signal = resample(signal, length(t), length(signal));
            
        case 'Zigbee'
            fc = 2.48e9; symbolRate = 250e3;
            chips = randi([0 1], 1, ceil(length(t)*symbolRate/config.sampleRate)*32);
            signal = pskmod(chips, 2, 0);
            signal = resample(signal, length(t), length(chips));
            
        case 'Radar'
            fc = 10e9; B = 100e6; T = config.duration;
            signal = chirp(t, fc-B/2, T, fc+B/2);
            
        case 'LTE'
            fc = 800e6; nSubcarriers = 600; nSymbols = 14;
            data = randi([0 3], nSubcarriers, nSymbols);
            lteSig = qammod(data(:), 16);
            lteSig = reshape(lteSig, nSubcarriers, nSymbols);
            signal = ifft(lteSig, nSubcarriers);
            signal = signal(:)';
            signal = resample(signal, length(t), length(signal));
            
        case 'Noise'
            signal = randn(size(t)) + 1j*randn(size(t));
            
        otherwise
            signal = zeros(size(t));
    end
end

function signal = applyImpairments(signal, envParams, config)
    t = (0:length(signal)-1) / config.sampleRate;
    
    % IQ Imbalance
    iqImb = envParams.iqImbalance * (rand - 0.5) * 2;
    gainImb = 1 + iqImb;
    phaseImb = iqImb * pi / 180;
    
    I = real(signal);
    Q = imag(signal);
    I_imb = gainImb * I;
    Q_imb = cos(phaseImb) * Q + sin(phaseImb) * I;
    signal = I_imb + 1j * Q_imb;
    
    % DC Offset
    dcI = envParams.dcOffset * (rand - 0.5);
    dcQ = envParams.dcOffset * (rand - 0.5);
    signal = signal + (dcI + 1j*dcQ);
    
    % Phase Noise
    phaseNoisePower = envParams.phaseNoise;
    phaseNoiseStd = sqrt(10^(phaseNoisePower/10));
    phaseNoise = cumsum(randn(size(signal)) * phaseNoiseStd);
    signal = signal .* exp(1j * phaseNoise);
    
    % Frequency Offset
    cfoError = envParams.cfoError * (rand - 0.5) * 2;
    signal = signal .* exp(1j * 2 * pi * cfoError * t);
    
    % Quantization (12-bit ADC)
    nBits = 12;
    maxVal = max(abs(signal));
    if maxVal > 0
        signal = signal / maxVal;
        quantLevels = 2^nBits;
        signal = round(signal * quantLevels/2) / (quantLevels/2);
        signal = signal * maxVal;
    end
end