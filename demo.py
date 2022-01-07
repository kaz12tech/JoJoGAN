import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import lpips
import wandb
from model import *
from e4e_projection import projection as e4e_projection
import subprocess

from copy import deepcopy
import argparse
import glob

os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('test_output', exist_ok=True)

os.makedirs('train_aligned', exist_ok=True)
os.makedirs('train_inversion', exist_ok=True)
os.makedirs('train_style_images', exist_ok=True)

drive_ids = {
    "stylegan2-ffhq-config-f.pt": "1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK",
    "e4e_ffhq_encode.pt": "1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7",
    "restyle_psp_ffhq_encode.pt": "1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd",
    "arcane_caitlyn.pt": "1gOsDTiTPcENiFOrhmkkxJcTURykW1dRc",
    "arcane_caitlyn_preserve_color.pt": "1cUTyjU-q98P75a8THCaO545RTwpVV-aH",
    "arcane_jinx_preserve_color.pt": "1jElwHxaYPod5Itdy18izJk49K1nl4ney",
    "arcane_jinx.pt": "1quQ8vPjYpUiXM4k1_KIwP4EccOefPpG_",
    "arcane_multi_preserve_color.pt": "1enJgrC08NpWpx2XGBmLt1laimjpGCyfl",
    "arcane_multi.pt": "15V9s09sgaw-zhKp116VHigf5FowAy43f",
    "disney.pt": "1zbE2upakFUAx8ximYnLofFwfT8MilqJA",
    "disney_preserve_color.pt": "1Bnh02DjfvN_Wm8c4JdOiNV4q9J7Z_tsi",
    "jojo.pt": "13cR2xjIBj8Ga5jMO7gtxzIJj2PDsBYK4",
    "jojo_preserve_color.pt": "1ZRwYLRytCEKi__eT2Zxv1IlV6BGVQ_K2",
    "jojo_yasuho.pt": "1grZT3Gz1DLzFoJchAmoj3LoM9ew9ROX_",
    "jojo_yasuho_preserve_color.pt": "1SKBu1h0iRNyeKBnya_3BBmLr4pkPeg_L",
    "supergirl.pt": "1L0y9IYgzLNzB-33xTpXpecsKU-t9DpVC",
    "supergirl_preserve_color.pt": "1VmKGuvThWHym7YuayXxjv0fSn32lfDpE",
    "art.pt": "1a0QDEHwXQ6hE_FcYEyNMuv5r5UnRQLKT",
}

# モデルダウンロード
for key in drive_ids:
    file_id = drive_ids[key]
    dst_path = './models/' + key
    if not os.path.exists(dst_path):
        res = subprocess.call(['gdown', '--id', file_id, '-O', dst_path])
print('Model download completed')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
latent_dim = 512
print('The device used for torch is', device)

transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def get_args():
    parser = argparse.ArgumentParser()
    # スタイルの適用に利用する画像を指定
    parser.add_argument("--img_path", type=str, default='./test_input/sample_input.jpg')
    
    # 事前学習済みモデルを選択(art, arcane_multi, supergirl, arcane_jinx, arcane_caitlyn, jojo_yasuho, jojo, disney)
    parser.add_argument("--pretrained", type=str, default='arcane_multi', 
        help='please select art, arcane_multi, supergirl, arcane_jinx, arcane_caitlyn, jojo_yasuho, jojo, disney')
    
    # 事前学習済みモデル、及び、学習時にStyleの色を適用する場合はTrue
    parser.add_argument("--preserve_color", type=bool, default=True)
    
    # ランダムにサンプルにスタイルを適用。サンプルを指定 
    parser.add_argument("--n_sample", type=int, default=5)

    # seed
    parser.add_argument("--seed", type=int, default=1234)

    # 学習時に使用するstyleのディレクトリを指定
    parser.add_argument("--img_dir", type=str, default='./train_style_images/')

    # 学習時のiteration
    parser.add_argument("--num_iter", type=int, default=200)

    args = parser.parse_args()

    return (args)

def main():
    # generatorのロード
    original_generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    mean_latent = original_generator.mean_latent(10000)

    # finetune generatorのロード
    generator = deepcopy(original_generator)


    args = get_args()

    filepath = args.img_path
    name = strip_path_extension(filepath)+'.pt'

    # 顔部分をcrop
    aligned_face = align_face(filepath)

    my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

    aligned_face.save('./test_output/Aligned_face.jpg')

    # pretrain styleをロード
    pretrained = args.pretrained
    preserve_color = args.preserve_color

    if preserve_color:
        ckpt = f'{pretrained}_preserve_color.pt'
    else:
        ckpt = f'{pretrained}.pt'

    # モデルロード
    ckpt = torch.load(os.path.join('models', ckpt), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"], strict=False)

    n_sample = args.n_sample
    seed = args.seed

    # Style適用
    torch.manual_seed(seed)
    with torch.no_grad():
        generator.eval()
        z = torch.randn(n_sample, latent_dim, device=device)

        original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
        sample = generator([z], truncation=0.7, truncation_latent=mean_latent)

        original_my_sample = original_generator(my_w, input_is_latent=True)
        my_sample = generator(my_w, input_is_latent=True)

    # 生成結果保存
    if pretrained == 'arcane_multi':
        style_path = f'style_images_aligned/arcane_jinx.png'
    else:   
        style_path = f'style_images_aligned/{pretrained}.png'
    style_image = transform(Image.open(style_path)).unsqueeze(0).to(device)
    face = transform(aligned_face).unsqueeze(0).to(device)
    
    my_output = torch.cat([style_image, face, my_sample], 0)
    save_image( 
        utils.make_grid(my_output, normalize=True, range=(-1, 1)),
        './test_output/sample_output.png'
        )

    output = torch.cat([original_sample, sample], 0)
    save_image(
        utils.make_grid(output, normalize=True, range=(-1, 1), nrow=n_sample),
        './test_output/random_sample_output.png'
    )

    ###################################################################################
    # train
    ###################################################################################
    folderpath = args.img_dir
    images = glob.glob( os.path.join(folderpath, '*.*') )

    targets = []
    latents = []

    for image in images:
        style_path = image
        name = strip_path_extension(os.path.basename(image))

        # crop and align the face
        style_aligned_path = os.path.join('train_aligned', os.path.basename(image))
        if not os.path.exists(style_aligned_path):
            style_aligned = align_face(style_path)
            style_aligned.save(style_aligned_path)
        else:
            style_aligned = Image.open(style_aligned_path).convert('RGB')

        # GAN invert
        style_code_path = os.path.join('train_inversion', f'{name}.pt')
        if not os.path.exists(style_code_path):
            latent = e4e_projection(style_aligned, style_code_path, device)
        else:
            latent = torch.load(style_code_path)['latent']

        targets.append(transform(style_aligned).to(device))
        latents.append(latent.to(device))
    
    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)

    target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
    
    alpha =  1.0 #@param {type:"slider", min:0, max:1, step:0.1}
    alpha = 1-alpha
    preserve_color = args.preserve_color
    num_iter = args.num_iter

    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # reset generator
    del generator
    generator = deepcopy(original_generator)

    g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

    # Which layers to swap for generating a family of plausible real images -> fake image
    if preserve_color:
        id_swap = [7,9,11,15,16,17]
    else:
        id_swap = list(range(7, generator.n_latent))

    for idx in tqdm(range(num_iter)):
        if preserve_color:
            random_alpha = 0
        else:
            random_alpha = np.random.uniform(alpha, 1)
        mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = latents.clone()
        in_latent[:, id_swap] = alpha*latents[:, id_swap] + (1-alpha)*mean_w[:, id_swap]

        img = generator(in_latent, input_is_latent=True)
        loss = lpips_fn(F.interpolate(img, size=(256,256), mode='area'), F.interpolate(targets, size=(256,256), mode='area')).mean()

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()


    torch.manual_seed(seed)
    with torch.no_grad():
        generator.eval()
        z = torch.randn(n_sample, latent_dim, device=device)
    
        original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
        sample = generator([z], truncation=0.7, truncation_latent=mean_latent)
    
        original_my_sample = original_generator(my_w, input_is_latent=True)
        my_sample = generator(my_w, input_is_latent=True)
    
    # display reference images
    style_images = []
    for image in images:
        name = os.path.basename(image)
        #style_path = f'train_aligned/{strip_path_extension(name)}.png'
        style_path = os.path.join('train_aligned', name)
        style_image = transform(Image.open(style_path))
        style_images.append(style_image)
        
    face = transform(aligned_face).to(device).unsqueeze(0)
    style_images = torch.stack(style_images, 0).to(device)

    save_image(
        utils.make_grid(style_images, normalize=True, range=(-1, 1)),
        './train_aligned/trained_references.png'
    )

    my_output = torch.cat([face, my_sample], 0)
    save_image(
        utils.make_grid(my_output, normalize=True, range=(-1, 1)),
        './train_aligned/trained_sample.png'
    )

    output = torch.cat([original_sample, sample], 0)
    save_image(
        utils.make_grid(output, normalize=True, range=(-1, 1), nrow=n_sample),
        './train_aligned/trained_random_sample.png'
    )

# python3 demo.py \
# --img_path ./test_input/sample_input.jpg \
# --pretrained jojo \
# --preserve_color True \
# --n_sample 5 \
# --seed 1234 \
# --img_dir ./style_images/
if __name__ == '__main__':
    main()