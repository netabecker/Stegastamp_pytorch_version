import os
import yaml
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from easydict import EasyDict

"""
Run this test in order to check if the model was successful
in encoding and decoding the images
"""

with open('cfg/setting.yaml', 'r') as f:
    param = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

def main():
    enc_model = os.path.join(param.checkpoints_path, "encoder_best_secret_loss.pth")
    dec_model = os.path.join(param.checkpoints_path, "decoder_best_secret_loss.pth")
    save_dir = os.path.join("./image"+enc_model[27:-12])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_image = os.path.join("./small_images_dir")

    os.system("python encode_image.py "+enc_model+" --images_dir="+test_image+" --secret=abcdefg "+"--save_dir="+save_dir)
    os.system("python decode_image.py " + dec_model + " --images_dir=" + save_dir + " > " + save_dir + "/summary.txt")

if __name__ == "__main__":
    main()