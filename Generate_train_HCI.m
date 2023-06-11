%% Initialization
clear all;
clc;
%close all;
addpath(genpath('./Functions/'))


%% Parameters setting
%save = 'F:\LFASR\Data\';
%mkdir(save);
angRes = 2;
angRes_label = 8;
patchsize = 64;
stride = 16;
sourceDataPath = 'I:\LFASR\Dataset\Train_mat\synthetic_ori\';
sourceDatasets = dir(sourceDataPath);%获得指定文件夹下的所有子文件夹和文件
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);
idx = 0;
SavePath = ['./Data/TrainingData', '_HCIori_', num2str(angRes), 'x', num2str(angRes), '_', 'ASR', '_', num2str(angRes_label), 'x', num2str(angRes_label), '/'];
if exist(SavePath, 'dir')==0
    mkdir(SavePath);
end

for DatasetIndex = 1 : datasetsNum
    sourceDataFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/'];
    folders = dir(sourceDataFolder); % list the scenes
    if isempty(folders)
        continue
    end
    folders(1:2) = [];
    sceneNum = length(folders);
    
    for iScene = 1 : sceneNum
        idx_s = 0;
        sceneName = folders(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating training data of Scene_%s in Dataset %s......\t\t', sceneName, sourceDatasets(DatasetIndex).name);
        dataPath = [sourceDataFolder, folders(iScene).name];
        data = load(dataPath);
        
        LF = data.LF; %读取光场数据      
        LF = LF(:, :, :, :, 1:3);%5D
        [U, V, H, W, ~] = size(LF);
        
        ind_all = linspace(1,angRes_label*angRes_label,angRes_label*angRes_label);
        ind_all = reshape(ind_all,angRes_label,angRes_label)';
        delt = (angRes_label-1)/(angRes-1);
        ind = ind_all(1:delt:angRes_label,1:delt:angRes_label);

        for h = 1 : stride : H-patchsize+1
            for w = 1 : stride : W-patchsize+1                
                SAI = single(zeros(angRes_label, angRes_label, patchsize, patchsize));%
                data  = single(zeros(angRes*patchsize, angRes*patchsize));
                label = single(zeros(angRes_label*patchsize, angRes_label*patchsize));
                LFhr = LF(1:angRes_label, 1:angRes_label, h:h+patchsize-1, w:w+patchsize-1, :); % Extract central angRes*angRes views
                size(LFhr);
                LFlr = LFhr(1:delt:angRes_label,1:delt:angRes_label,:,:,:);
                count = 1;
                for u = 1 : angRes_label
                    for v = 1 : angRes_label
                        k = (u-1)*angRes_label + v;                        
                        SAItemp = squeeze(LFhr(u, v, :, :, :));
                        SAItemp = rgb2ycbcr(double(SAItemp));
                        temp = squeeze(SAItemp(:,:,1));%降低了维度，把维度上是1的去掉
                        label((u-1)*patchsize+1 : u*patchsize, (v-1)*patchsize+1 : v*patchsize) = temp;
                    end
                end
                for u = 1 : angRes
                    for v = 1 : angRes
                        k = (u-1)*angRes + v;                        
                        SAItemp = squeeze(LFlr(u, v, :, :, :));
                        %imwrite(SAItemp,[save 'ours_' num2str(u) '_' num2str(v) '.png']);%ours
                        SAItemp = rgb2ycbcr(double(SAItemp));
                        temp = squeeze(SAItemp(:,:,1));%降低了维度，把维度上是1的去掉
                        data((u-1)*patchsize+1 : u*patchsize, (v-1)*patchsize+1 : v*patchsize) = temp;
                    end
                end


                idx = idx + 1;
                SavePath_H5 = [SavePath, num2str(idx,'%06d'),'.h5'];
                h5create(SavePath_H5, '/data', size(data), 'Datatype', 'single');
                h5write(SavePath_H5, '/data', single(data), [1,1], size(data));
                h5create(SavePath_H5, '/label', size(label), 'Datatype', 'single');
                h5write(SavePath_H5, '/label', single(label), [1,1], size(label));       
            end
        end
        fprintf([num2str(idx), ' training samples have been generated\n']);
    end
end

