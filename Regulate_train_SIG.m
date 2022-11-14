clear; close all;

%% path
savepath = './Dataset/Train_mat/real/SIG/';
folder = 'I:/LFASR/Dataset/Training/SIG';

listname = 'I:/LFASR/Dataset/list/Train_SIG.txt';
f = fopen(listname);
if( f == -1 )
    error('%s does not exist!', listname);
end
C = textscan(f, '%s', 'CommentStyle', '#');
list = C{1};
fclose(f); 

%% params
H = 374;
W = 540;

allah = 14;
allaw = 14;

ah = 9;
aw = 9;

%% initialization
%LF = zeros(ah, aw, H, W,  3, 'single');

%% generate data
for k = 1:size(list,1)
    lfname = list{k};
    lf_path = sprintf('%s/%s.png',folder,lfname);
    disp(lf_path);
    
    eslf = im2single(imread(lf_path));
	%eslf = single(im2uint8(imread(lf_path)))/255;
    img = zeros(allah,allaw,H,W,3,'single');

    for v = 1 : allah
        for u = 1 : allah            
            sub = eslf(v:allah:end,u:allah:end,:);            
            %sub = rgb2ycbcr(sub);           
            img(v,u,:,:,:) = sub(1:H,1:W,:);        
        end
    end
        
    LF = img(4:11,4:11,:,:,:);

    %LF(:, :, :, :, count) = img;
    save_path = [savepath, lfname, '.mat'];
    save(save_path, 'LF'); 
end  
 

