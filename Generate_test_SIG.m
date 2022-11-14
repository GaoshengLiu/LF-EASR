%% Initialization
clear all;
clc;
addpath(genpath('./Functions/'))

%% Parameters setting

angRes = 2; % Angular Resolution, options, e.g., 3, 5, 7, 9. Default: 5
angRes_label = 5;

sourceDataPath = '.\Dataset\Test_mat\real\';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);
idx = 0;

for DatasetIndex = 1 : datasetsNum
    DatasetName = sourceDatasets(DatasetIndex).name;
    SavePath = ['./Data/TestData', '_SIG_' num2str(angRes), 'x', num2str(angRes), '_', 'ASR', '_', num2str(angRes_label), 'x', num2str(angRes_label),  '/', DatasetName];
    if exist(SavePath, 'dir')==0
        mkdir(SavePath);
    end
    
    sourceDataFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/'];
    folders = dir(sourceDataFolder); % list the scenes
    folders(1:2) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating test data of Scene_%s in Dataset %s......\n', sceneName, sourceDatasets(DatasetIndex).name);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        LF = data.LF;
        [U, V, H, W, ~] = size(LF);
        while mod(H, 2) ~= 0
            H = H - 1;
        end
        while mod(W, 2) ~= 0
            W = W - 1;
        end
        ind_all = linspace(1,angRes_label*angRes_label,angRes_label*angRes_label);
        ind_all = reshape(ind_all,angRes_label,angRes_label)';
        delt = (angRes_label-1)/(angRes-1);
        ind = ind_all(1:delt:angRes_label,1:delt:angRes_label);
        
        LF = LF(1:angRes_label, 1:angRes_label, 1:H, 1:W, 1:3);
        [~, ~, H_, W_, ~] = size(LF);
        LFlr = LF(1:delt:angRes_label,1:delt:angRes_label,:,:,:);
        [U, V, H, W, ~] = size(LF);
        label = single(zeros(U*H, V*W));

        for u = 1 : U
            for v = 1 : V
                SAI_rgb = squeeze(LF(u, v, :, :, :));
                SAI_ycbcr = rgb2ycbcr(double(SAI_rgb));
                label((u-1)*H+1 : u*H, (v-1)*W+1 : v*W) = SAI_ycbcr(:, :, 1);
            end
        end
        
        
        [Ul, Vl, Hl, Wl, ~] = size(LFlr);
        data = single(zeros(Ul*Hl, Vl*Wl));
        for u = 1 : Ul
            for v = 1 : Vl
                SAI_rgb = squeeze(LFlr(u, v, :, :, :));
                SAI_ycbcr = rgb2ycbcr(double(SAI_rgb));
                data((u-1)*Hl+1 : u*Hl, (v-1)*Wl+1 : v*Wl) = SAI_ycbcr(:, :, 1);
            end
        end
  
        SavePath_H5 = [SavePath, '/', sceneName, '.h5'];
        h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
        h5write(SavePath_H5, '/data', single(data), [1,1], size(data));
        h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
        h5write(SavePath_H5, '/label', single(label), [1,1], size(label));
        idx = idx + 1;
    end
end


