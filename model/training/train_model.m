%% Fixed ResNet-18 Training for 10K Dataset

clear; close all; clc;

%% Configuration
config.dataDirectory = 'C:\Users\derri\OneDrive\Dokumente\MATLAB\RF_10K_Optimized_Dataset';
config.modelSavePath = 'C:\Users\derri\OneDrive\Dokumente\MATLAB';
config.modelName = 'resnet18_10k_model.mat';

% Training splits
config.trainingSplit = 0.70;
config.validationSplit = 0.15;
config.testSplit = 0.15;

% Training parameters
config.maxEpochs = 20;              % 20 epochs for 10K data
config.miniBatchSize = 16;
config.initialLearnRate = 1e-4;
config.learnRateDropPeriod = 7;
config.learnRateDropFactor = 0.5;
config.validationFrequency = 100;
config.l2Regularization = 0.0001;

% Memory management
config.rgbConversionBatchSize = 1000;

fprintf('========================================================\n');
fprintf('  RESNET-18 TRAINING - 10K DATASET\n');
fprintf('========================================================\n');
fprintf('Data directory: %s\n', config.dataDirectory);
fprintf('Model: %s\n', config.modelName);
fprintf('Max epochs: %d\n', config.maxEpochs);
fprintf('Mini-batch size: %d\n\n', config.miniBatchSize);

%% Step 1: Load Metadata
fprintf('Step 1: Loading metadata...\n');

metadataFile = fullfile(config.dataDirectory, 'dataset_metadata.mat');
if ~exist(metadataFile, 'file')
    error('Metadata not found: %s\nRun generate_10k_optimized first!', metadataFile);
end

metadata = load(metadataFile);
signalTypes = metadata.metadata.signalTypes;
numClasses = length(signalTypes);

fprintf('  Signal types: %s\n', strjoin(signalTypes, ', '));
fprintf('  Expected samples: %d\n\n', metadata.metadata.totalSamples);

%% Step 2: Load Batch Files
fprintf('Step 2: Loading batch files...\n');

% Look for 10K batch files (correct naming)
batchFiles = dir(fullfile(config.dataDirectory, 'rf_10k_batch_*.mat'));

if isempty(batchFiles)
    error('No batch files found in: %s\nRun generate_10k_optimized first!', config.dataDirectory);
end

fprintf('  Found %d batches\n', length(batchFiles));

% Pre-allocate arrays
allData = [];
allLabels = [];

% Load all batches
for i = 1:length(batchFiles)
    batchPath = fullfile(config.dataDirectory, batchFiles(i).name);
    batch = load(batchPath);
    
    % Use correct field names from 10K generator
    if isfield(batch, 'saveBatchData')
        batchData = batch.saveBatchData;
        batchLabels = batch.saveBatchLabels;
    else
        error('Unexpected batch structure in file: %s', batchFiles(i).name);
    end
    
    % Concatenate
    if isempty(allData)
        allData = batchData;
        allLabels = batchLabels;
    else
        allData = cat(3, allData, batchData);
        allLabels = [allLabels; batchLabels]; %#ok<AGROW>
    end
    
    if mod(i, 20) == 0 || i == length(batchFiles)
        fprintf('  Loaded %d/%d batches (%.1f%%)...\n', i, length(batchFiles), 100*i/length(batchFiles));
    end
end

totalSamples = size(allData, 3);
fprintf('  ✓ Loaded %d samples\n', totalSamples);
fprintf('  Data: [%d × %d × %d]\n', size(allData,1), size(allData,2), size(allData,3));
fprintf('  Labels: [%d × %d]\n', size(allLabels,1), size(allLabels,2));

% Verify data integrity
if size(allLabels, 1) ~= totalSamples
    error('Data mismatch: %d spectrograms but %d label rows!', totalSamples, size(allLabels,1));
end
fprintf('  ✓ Data and labels match\n\n');

%% Step 3: Split Data
fprintf('Step 3: Splitting dataset...\n');

rng(42);
idx = randperm(totalSamples);

nTrain = round(config.trainingSplit * totalSamples);
nVal = round(config.validationSplit * totalSamples);

trainIdx = idx(1:nTrain);
valIdx = idx(nTrain+1:nTrain+nVal);
testIdx = idx(nTrain+nVal+1:end);

fprintf('  Total: %d samples\n', totalSamples);
fprintf('  Training: %d samples\n', length(trainIdx));
fprintf('  Validation: %d samples\n', length(valIdx));
fprintf('  Test: %d samples\n\n', length(testIdx));

% Split labels
y_train = allLabels(trainIdx, :);
y_val = allLabels(valIdx, :);
y_test = allLabels(testIdx, :);

%% Step 4: Convert to RGB in Batches
fprintf('Step 4: Converting to RGB (memory efficient)...\n');
fprintf('Processing %d samples at a time\n\n', config.rgbConversionBatchSize);

targetSize = [224, 224];

% Convert training data
fprintf('  Converting training data (%d samples)...\n', length(trainIdx));
X_train = convertBatchToRGB(allData, trainIdx, targetSize, config.rgbConversionBatchSize);
fprintf('    ✓ Training: [%d × %d × %d × %d]\n', size(X_train));

% Convert validation data
fprintf('  Converting validation data (%d samples)...\n', length(valIdx));
X_val = convertBatchToRGB(allData, valIdx, targetSize, config.rgbConversionBatchSize);
fprintf('    ✓ Validation: [%d × %d × %d × %d]\n', size(X_val));

% Convert test data
fprintf('  Converting test data (%d samples)...\n', length(testIdx));
X_test = convertBatchToRGB(allData, testIdx, targetSize, config.rgbConversionBatchSize);
fprintf('    ✓ Test: [%d × %d × %d × %d]\n\n', size(X_test));

% Clear to save memory
clear allData allLabels idx batchFiles batch;
fprintf('  ✓ Original data cleared\n\n');

%% Step 5: Load ResNet-18
fprintf('Step 5: Loading ResNet-18...\n');

try
    net = resnet18;
    fprintf('  ✓ ResNet-18 loaded\n');
    fprintf('  Input: [%d %d %d]\n', net.Layers(1).InputSize);
    fprintf('  Total layers: %d\n\n', length(net.Layers));
catch ME
    error('ResNet-18 not available:\n%s\nInstall from Add-Ons menu.', ME.message);
end

%% Step 6: Modify Network
fprintf('Step 6: Modifying network for multi-label...\n');

lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000', 'prob', 'ClassificationLayer_predictions'});

newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_multilabel', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    sigmoidLayer('Name', 'sigmoid')
    regressionLayer('Name', 'output')
];

lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'fc_multilabel');

fprintf('  ✓ Modified for %d-class multi-label\n', numClasses);
fprintf('  Output: Sigmoid + Regression\n\n');

%% Step 7: Configure Training
fprintf('Step 7: Configuring training...\n');

options = trainingOptions('adam', ...
    'MaxEpochs', config.maxEpochs, ...
    'MiniBatchSize', config.miniBatchSize, ...
    'InitialLearnRate', config.initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', config.learnRateDropPeriod, ...
    'LearnRateDropFactor', config.learnRateDropFactor, ...
    'L2Regularization', config.l2Regularization, ...
    'ValidationData', {X_val, y_val}, ...
    'ValidationFrequency', config.validationFrequency, ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu', ...
    'Shuffle', 'every-epoch');

fprintf('  ✓ Training configured\n');
fprintf('  Optimizer: Adam\n');
fprintf('  Learning rate: %.5f\n', config.initialLearnRate);
fprintf('  Validation frequency: every %d iterations\n\n', config.validationFrequency);

%% Step 8: Train Network
fprintf('========================================\n');
fprintf('Step 8: Training network\n');
fprintf('========================================\n');
fprintf('Expected time: 18-22 hours on 2-core CPU\n');
fprintf('Started: %s\n\n', datestr(now));

tic;
try
    trainedNet = trainNetwork(X_train, y_train, lgraph, options);
    trainingTime = toc;
    
    fprintf('\n✓ Training complete!\n');
    fprintf('  Time: %.2f hours\n', trainingTime/3600);
    fprintf('  Finished: %s\n\n', datestr(now));
catch ME
    fprintf('\n❌ Training failed!\n');
    fprintf('Error: %s\n', ME.message);
    rethrow(ME);
end

%% Step 9: Evaluate
fprintf('Step 9: Evaluating on test set...\n');

y_pred = predict(trainedNet, X_test);
y_pred_binary = double(y_pred > 0.5);

fprintf('\n========================================\n');
fprintf('PER-CLASS PERFORMANCE\n');
fprintf('========================================\n');
fprintf('%-12s %8s %8s %8s %8s\n', 'Signal', 'Accuracy', 'Prec', 'Recall', 'F1');
fprintf('--------------------------------------------------------\n');

classMetrics = zeros(numClasses, 4);

for i = 1:numClasses
    tp = sum(y_pred_binary(:,i) == 1 & y_test(:,i) == 1);
    fp = sum(y_pred_binary(:,i) == 1 & y_test(:,i) == 0);
    fn = sum(y_pred_binary(:,i) == 0 & y_test(:,i) == 1);
    tn = sum(y_pred_binary(:,i) == 0 & y_test(:,i) == 0);
    
    accuracy = (tp + tn) / (tp + fp + fn + tn);
    precision = tp / (tp + fp + eps);
    recall = tp / (tp + fn + eps);
    f1 = 2 * precision * recall / (precision + recall + eps);
    
    classMetrics(i, :) = [accuracy, precision, recall, f1];
    
    fprintf('%-12s %7.1f%% %7.1f%% %7.1f%% %7.2f\n', ...
        signalTypes{i}, accuracy*100, precision*100, recall*100, f1);
end

fprintf('--------------------------------------------------------\n');
fprintf('%-12s %7.1f%% %7.1f%% %7.1f%% %7.2f\n', 'AVERAGE', ...
    mean(classMetrics(:,1))*100, mean(classMetrics(:,2))*100, ...
    mean(classMetrics(:,3))*100, mean(classMetrics(:,4)));

hammingAcc = sum(all(y_pred_binary == y_test, 2)) / size(y_test, 1);
fprintf('\nExact match accuracy: %.1f%%\n', hammingAcc*100);
fprintf('========================================\n\n');

%% Step 10: Save Model
fprintf('Step 10: Saving model...\n');

modelPath = fullfile(config.modelSavePath, config.modelName);
save(modelPath, 'trainedNet', 'signalTypes', 'classMetrics', 'config', '-v7.3');
fprintf('  ✓ Model saved: %s\n', modelPath);

resultsPath = fullfile(config.modelSavePath, 'training_results_10k.mat');
results = struct('signalTypes', signalTypes, 'classMetrics', classMetrics, ...
    'hammingAccuracy', hammingAcc, 'trainingTime', trainingTime, ...
    'trainedDate', datestr(now));
save(resultsPath, 'results');
fprintf('  ✓ Results saved: %s\n\n', resultsPath);

%% Summary
fprintf('========================================\n');
fprintf('TRAINING COMPLETE!\n');
fprintf('========================================\n');
fprintf('Model: %s\n', config.modelName);
fprintf('Training time: %.2f hours\n', trainingTime/3600);
fprintf('Average F1: %.2f\n', mean(classMetrics(:,4)));
fprintf('Exact match: %.1f%%\n', hammingAcc*100);
fprintf('\nBest: %s (F1=%.2f)\n', signalTypes{classMetrics(:,4)==max(classMetrics(:,4))}, max(classMetrics(:,4)));
fprintf('Worst: %s (F1=%.2f)\n', signalTypes{classMetrics(:,4)==min(classMetrics(:,4))}, min(classMetrics(:,4)));
fprintf('========================================\n');

%% Helper Function
function rgbData = convertBatchToRGB(allData, indices, targetSize, batchSize)
    numSamples = length(indices);
    numBatches = ceil(numSamples / batchSize);
    
    rgbData = zeros(targetSize(1), targetSize(2), 3, numSamples, 'single');
    
    for b = 1:numBatches
        startIdx = (b-1) * batchSize + 1;
        endIdx = min(b * batchSize, numSamples);
        batchIndices = indices(startIdx:endIdx);
        batchSize_actual = length(batchIndices);
        
        for i = 1:batchSize_actual
            spec = allData(:, :, batchIndices(i));
            specResized = imresize(spec, targetSize);
            specNorm = mat2gray(specResized);
            specRGB = ind2rgb(uint8(specNorm * 255), jet(256));
            
            globalIdx = startIdx + i - 1;
            rgbData(:, :, :, globalIdx) = single(specRGB);
        end
        
        if mod(b, 5) == 0 || b == numBatches
            fprintf('    Batch %d/%d (%.1f%%)\n', b, numBatches, 100*b/numBatches);
        end
    end
end