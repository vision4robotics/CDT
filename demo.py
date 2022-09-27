
import torch
import torchvision
import torch.optim
import os
from basicsr.models.archs.CDT_arch import CDT
from denoiser.denoiser_builder import Denoiser
import argparse
from PIL import Image


torch.set_num_threads(1) 

parser = argparse.ArgumentParser(description='demo processer')
parser.add_argument('--d_weights', default='./experiments/CDT/model.pth', type=str,
                    help='weights')
args = parser.parse_args()


if __name__ == '__main__':

    parameters = {'CDT':{'in_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[1,2,3,4], 'heads':[1,2,2,4], 'ffn_expansion_factor':2.67, 'bias':False, 'LayerNorm_type':'BiasFree'},
                  }

    model = CDT(**parameters['CDT'])
    cdt = Denoiser(model, args)

    filePath = './CDT/demo/img_ori/'

    file_list = os.listdir(filePath)

    with torch.no_grad():

        for file_name in file_list:
            img = Image.open(filePath+file_name)
            restored = cdt.single_denoise(img)
            dns_path = filePath.replace('img_ori', 'img_dns')
            torchvision.utils.save_image(restored, dns_path+file_name)


        

