numFrames = 1600;
signalLabels = containers.Map(...
        {'WLAN', 'ZigBee', 'Bluetooth', 'SmartBAN'}, ...
        [16, 32, 64, 128]);  % DO NOT CHANGE THIS FIELD!
sr = 80e6;  % DO NOT CHANGE THIS FIELD!
imageSize = {[256,256]}; % DO NOT CHANGE THIS FIELD!
useGPU = true;
wantInterf = true;
wantPlot = false;
creatingTrainingImages(numFrames, signalLabels, sr, imageSize, useGPU, wantInterf, wantPlot)