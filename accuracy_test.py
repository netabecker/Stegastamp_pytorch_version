import os
import yaml
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from easydict import EasyDict


with open('cfg/setting.yaml', 'r') as f:
    param = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default=os.path.join(param.saved_models, "encoder.pth"))
    parser.add_argument('--decoder', type=str, default=os.path.join(param.saved_models, "decoder.pth"))
    parser.add_argument('--recompile', type=str, default='y')
    accuracy_args = parser.parse_args()

    # enc_model = os.path.join(param.saved_models, "encoder.pth")
    # dec_model = os.path.join(param.saved_models, "decoder.pth")
    save_dir = os.path.join("./image"+accuracy_args.encoder[14:-12])

    if 'y' in accuracy_args.recompile:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        test_image = os.path.join("./small_images_dir")

        os.system("python encode_image.py "+accuracy_args.encoder+" --images_dir="+test_image+" --secret=abcdefg "+"--save_dir="+save_dir)
        # # for filename in glob.glob(os.path.join(save_dir,"*jpg")):
        # #     # returned_value = os.system("decode_image.py "+dec_model+" --image="+filename)
        # #     if returned_value != 'Failed to decode'
        # #     print(filename)
        #
        os.system("python decode_image.py " + accuracy_args.decoder + " --images_dir=" + save_dir + " > " + save_dir + "/summary.txt")

    fail_count = 0
    file_location = save_dir + "/summary.txt"
    full_path = os.getcwd() + file_location[1:]

    with open(full_path, "a+"):
        for line in full_path:
            if "Failed to decode" in line:
                fail_count = fail_count + 1

        print('-------------------------------------------')
        print(f'Fail number: {fail_count} out of 21')
        print('-------------------------------------------')



if __name__ == "__main__":
    main()