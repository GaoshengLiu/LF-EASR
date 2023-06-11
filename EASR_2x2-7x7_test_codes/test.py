import time
import argparse
import scipy.misc
import torch.backends.cudnn as cudnn
from utils import *
from model_light27 import Net
from tqdm import tqdm
import scipy.io as sio
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=2, help="angular resolution")
    parser.add_argument("--angout", type=int, default=7, help="angular resolution")
    parser.add_argument("--upscale_factor", type=int, default=7, help="upscale factor") ###for 2x2->7x7 factor is not the upsampling scale
    parser.add_argument('--testset_dir', type=str, default='I:/LFASR/Data/TestData_SIG_2x2_ASR_7x7/')

    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument("--stride", type=int, default=32, help="The stride between two test patches is set to patchsize/2")

    parser.add_argument('--model_path', type=str, default='./log/EASR_SIG.pth')
    parser.add_argument('--save_path', type=str, default='./Results/')

    return parser.parse_args()


def test(cfg, test_Names, test_loaders):

    net = Net(cfg.angin, cfg.angout, cfg.upscale_factor)
    net.to(cfg.device)
    cudnn.benchmark = True

    ##### get input index ######         
    ind_all = np.arange(cfg.angout*cfg.angout).reshape(cfg.angout, cfg.angout)        
    delt = (cfg.angout-1) // (cfg.angin-1)
    ind_source = ind_all[0:cfg.angout:delt, 0:cfg.angout:delt]
    #ind_source = np.array([18, 21, 42, 45])
    #ind_source = np.array([9, 14, 49, 54])
    ind_source = torch.from_numpy(ind_source.reshape(-1))

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:1': cfg.device})
        net.load_state_dict(model)#model.state_dict())['state_dict']
    else:
        print("=> no model found at '{}'".format(cfg.model_path))

    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            outLF, psnr_epoch_test, ssim_epoch_test = inference(test_loader, test_name, net, ind_source)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass
        pass


def inference(test_loader, test_name, net, ind_source):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
        label = label.squeeze()
        #print(data.shape)

        uh, vw = data.shape
        h0, w0 = uh // cfg.angin, vw // cfg.angin
        subLFin = LFdivide(data, cfg.angin, cfg.patchsize, cfg.stride)  # numU, numV, h*angin, w*angin
        numU, numV, H, W = subLFin.shape
        subLFout = torch.zeros(numU, numV, cfg.angout * cfg.patchsize, cfg.angout * cfg.patchsize)
        
        #s = time.time()

        for u in range(numU):
            for v in range(numV):
                tmp = subLFin[u, v, :, :].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    out = net(tmp.to(cfg.device))
                    subLFout[u, v, :, :] = out.squeeze()

        outLF = LFintegrate(subLFout, cfg.angout, cfg.patchsize, cfg.stride, h0, w0)
        # print(outLF.shape)
        # print(label.shape)
        # print(torch.max(outLF))
        # print(torch.max(label))

        psnr, ssim = cal_metrics(label, outLF, cfg.angout, ind_source)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name)

        sio.savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                        {'LF': outLF.numpy()})
        pass


    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return outLF, psnr_epoch_test, ssim_epoch_test


def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    test(cfg, test_Names, test_Loaders)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
