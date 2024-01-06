from PIL import Image
import numpy as np
import os
import torch
import argparse
import cv2
import torchvision.transforms.functional as tf
import torchvision
import torch.nn.functional as f
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm

"""parsing and configuration"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataroot', type=str, default='', help='dataset_dir')#/home/data/juliuxue/data/iHarmony4_data/
    parser.add_argument('--result_root', type=str, default='', help='dataset_dir')#/home/data/juliuxue/code/HDNet-main/HDNet-main/evaluate/110/results/
    parser.add_argument('--dataset_name', type=str, default='ihd', help='dataset_name')#    IHD
    parser.add_argument('--evaluation_type', type=str, default="our", help='evaluation type') #our
    parser.add_argument('--ssim_window_size', type=int, default=11, help='ssim window size')

    return parser.parse_args()


def main(dataset_name=None):
    cuda = True if torch.cuda.is_available() else False
    IMAGE_SIZE = np.array([256, 256])
    opt.dataset_name = dataset_name
    files = opt.dataroot + opt.dataset_name + '/' + opt.dataset_name + '_' + opt.phase + '.txt'
    if dataset_name == 'IHD':
        files = opt.dataroot + opt.dataset_name + '_' + opt.phase + '.txt'

    comp_paths = []
    harmonized_paths = []
    mask_paths = []
    real_paths = []

    with open(files, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()

            real_path = ''
            mask_path = ''
            comp_path = ''
            harmonized_path = ''
            if dataset_name == 'IHD':
                comp_path = os.path.join(opt.dataroot, line)
                name_str = line.split('/')#名字
                name_parts = line.split('_')
                mask_path = line.replace('composite_images', 'masks')
                mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')
                mask_path=os.path.join(opt.dataroot, mask_path)
                real_path =line.replace('composite_images', 'real_images')
                real_path = real_path.replace('_' + name_parts[-2] + '_' + name_parts[-1], '.jpg')
                real_path=os.path.join(opt.dataroot, real_path)
                harmonized_path=os.path.join(opt.result_root, name_str[0],name_str[-1])





            # if opt.evaluation_type == 'our':
            #     harmonized_path = os.path.join(opt.result_root, name_str)
            #     if os.path.exists(harmonized_path):
            #         real_path = os.path.join(opt.result_root, name_str.replace(".jpg", "_real.jpg"))
            #         mask_path = os.path.join(opt.result_root, name_str.replace(".jpg", "_mask.jpg"))
            #         comp_path = os.path.join(opt.result_root, name_str.replace(".jpg", "_comp.jpg"))
            # elif opt.evaluation_type == 'ori':
            #     comp_path = os.path.join(opt.dataroot, 'composite_images', line.rstrip())
            #     harmonized_path = comp_path
            #     if os.path.exists(comp_path):
            #         real_path = os.path.join(opt.dataroot, 'real_images', line.rstrip())
            #         name_parts = real_path.split('_')
            #         real_path = real_path.replace(('_' + name_parts[-2] + '_' + name_parts[-1]), '.jpg')
            #         mask_path = os.path.join(opt.dataroot, 'masks', line.rstrip())
            #         mask_path = mask_path.replace(('_' + name_parts[-1]), '.png')

            real_paths.append(real_path)
            mask_paths.append(mask_path)
            comp_paths.append(comp_path)
            harmonized_paths.append(harmonized_path)
    count = 0

    mse_scores = 0
    sk_mse_scores = 0
    fmse_scores = 0
    psnr_scores = 0
    fpsnr_scores = 0
    ssim_scores = 0
    fssim_scores = 0
    fore_area_count = 0
    fmse_score_list = []
    image_size = 256
    ##################################
    mse_fg1 = []
    mse_fg2 = []
    mse_fg3 = []
    mse_fgall = []
    psnr_fg1 = []
    psnr_fg2 = []
    psnr_fg3 = []
    psnr_fgall = []
    fmse_fg1 = []
    fmse_fg2 = []
    fmse_fg3 = []
    fmse_fgall = []
    fpsnr_fg1 = []
    fpsnr_fg2 = []
    fpsnr_fg3 = []
    fpsnr_fgall = []
    #################

    for i, harmonized_path in enumerate(tqdm(harmonized_paths)):
        count += 1

        harmonized = Image.open(harmonized_path).convert('RGB')
        real = Image.open(real_paths[i]).convert('RGB')
        mask = Image.open(mask_paths[i]).convert('1')
        if mask.size[0] != image_size:
            harmonized = tf.resize(harmonized, [image_size, image_size])
            mask = tf.resize(mask, [image_size, image_size])
            real = tf.resize(real, [image_size, image_size])

        harmonized_np = np.array(harmonized, dtype=np.float32)  # 范围是0-255
        real_np = np.array(real, dtype=np.float32)# 256*256*3
        mask = np.array(mask, dtype=np.uint8)

        harmonized = tf.to_tensor(harmonized_np).unsqueeze(0).cuda()
        real = tf.to_tensor(real_np).unsqueeze(0).cuda()# 3*256*256
        mask = tf.to_tensor(mask).unsqueeze(0).cuda()

        mse_score = mse(harmonized_np, real_np)
        psnr_score = psnr(real_np, harmonized_np, data_range=255)

        fore_area = torch.sum(mask)
        fmse_score = torch.nn.functional.mse_loss(harmonized * mask, real * mask) * 256 * 256 / fore_area

        mse_score = mse_score.item()
        fmse_score = fmse_score.item()
        fore_area_count += fore_area.item()
        fpsnr_score = 10 * np.log10((255 ** 2) / fmse_score)


        psnr_scores += psnr_score
        mse_scores += mse_score
        fmse_scores += fmse_score
        fpsnr_scores += fpsnr_score
        fg_ratio = fore_area / (256 * 256)
        if fg_ratio <= 0.05:
            mse_fg1.append(mse_score)
            psnr_fg1.append(psnr_score)
            fmse_fg1.append(fmse_score)
            fpsnr_fg1.append(fpsnr_score)
        elif fg_ratio <= 0.15:
            mse_fg2.append(mse_score)
            psnr_fg2.append(psnr_score)
            fmse_fg2.append(fmse_score)
            fpsnr_fg2.append(fpsnr_score)
        else:
            mse_fg3.append(mse_score)
            psnr_fg3.append(psnr_score)
            fmse_fg3.append(fmse_score)
            fpsnr_fg3.append(fpsnr_score)
        mse_fgall.append(mse_score)
        psnr_fgall.append(psnr_score)
        fmse_fgall.append(fmse_score)
        fpsnr_fgall.append(fpsnr_score)

        image_name = harmonized_path.split("/")
        image_fmse_info = (
        image_name[-1], round(fmse_score, 2), fore_area.item(), round(mse_score, 2), round(psnr_score, 2),
        round(fpsnr_scores, 4))
        fmse_score_list.append(image_fmse_info)

    mse_scores_mu = mse_scores / count
    psnr_scores_mu = psnr_scores / count
    fmse_scores_mu = fmse_scores / count
    fpsnr_scores_mu = fpsnr_scores / count
    ssim_scores_mu = ssim_scores / count
    fssim_score_mu = fssim_scores / count

    print(count)
    mean_sore = "%s MSE %0.2f | PSNR %0.2f | SSIM %0.4f |fMSE %0.2f | fPSNR %0.2f | fSSIM %0.4f" % (
    opt.dataset_name, mse_scores_mu, psnr_scores_mu, ssim_scores_mu, fmse_scores_mu, fpsnr_scores_mu, fssim_score_mu)
    print(mean_sore)
    mse_fg1 = np.array(mse_fg1).astype(np.float64)
    psnr_fg1 = np.array(psnr_fg1).astype(np.float64)
    fmse_fg1 = np.array(fmse_fg1).astype(np.float64)
    fpsnr_fg1 = np.array(fpsnr_fg1).astype(np.float64)
    print(len(mse_fg1), '%s: MSE %0.2f/PSNR %0.2f/fMSE %0.2f/fPSNR %0.2f' % (
        " |0.00-0.05: ", np.mean(mse_fg1), np.mean(psnr_fg1), np.mean(fmse_fg1), np.mean(fpsnr_fg1)))

    mse_fg2 = np.array(mse_fg2).astype(np.float64)
    psnr_fg2 = np.array(psnr_fg2).astype(np.float64)
    fmse_fg2 = np.array(fmse_fg2).astype(np.float64)
    fpsnr_fg2 = np.array(fpsnr_fg2).astype(np.float64)
    print(len(mse_fg2), '%s: MSE %0.2f/PSNR %0.2f/fMSE %0.2f/fPSNR %0.2f' % (
        " |0.05-0.15: ", np.mean(mse_fg2), np.mean(psnr_fg2), np.mean(fmse_fg2), np.mean(fpsnr_fg2)))

    mse_fg3 = np.array(mse_fg3).astype(np.float64)
    psnr_fg3 = np.array(psnr_fg3).astype(np.float64)
    fmse_fg3 = np.array(fmse_fg3).astype(np.float64)
    fpsnr_fg3 = np.array(fpsnr_fg3).astype(np.float64)
    print(len(mse_fg3), '%s: MSE %0.2f/PSNR %0.2f/fMSE %0.2f/fPSNR %0.2f' % (
        " |0.15-1.00: ", np.mean(mse_fg3), np.mean(psnr_fg3), np.mean(fmse_fg3), np.mean(fpsnr_fg3)))

    mse_fgall = np.array(mse_fgall).astype(np.float64)
    psnr_fgall = np.array(psnr_fgall).astype(np.float64)
    fmse_fgall = np.array(fmse_fgall).astype(np.float64)
    fpsnr_fgall = np.array(fpsnr_fgall).astype(np.float64)
    print(len(mse_fgall), '%s: MSE %0.2f/PSNR %0.2f/fMSE %0.2f/fPSNR %0.2f' % (
        " |0.00-1.0: ", np.mean(mse_fgall), np.mean(psnr_fgall), np.mean(fmse_fgall), np.mean(fpsnr_fgall)))

    return mse_scores_mu, fmse_scores_mu, psnr_scores_mu, fpsnr_scores_mu


def generstr(dataset_name='ALL'):
    datasets = ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night', 'IHD']
    if dataset_name == 'newALL':
        datasets = ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night', 'HVIDIT', 'newIHD']
    for i, item in enumerate(datasets):
        print(item)
        mse_scores_mu, fmse_scores_mu, psnr_scores_mu, fpsnr_scores_mu = main(dataset_name=item)


if __name__ == '__main__':
    opt = parse_args()
    if opt is None:
        exit()
    if opt.dataset_name == "ALL":
        generstr()
    elif opt.dataset_name == "newALL":
        generstr(dataset_name='newALL')
    else:
        main(dataset_name=opt.dataset_name)