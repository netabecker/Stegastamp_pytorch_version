from aux_functions import *
import os
import glob
import bchlib
import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import transforms

infoMessage(getLineNumber(), 'NOTE! The residual depicts the difference between the original image and the StegaStamp')
BCH_POLYNOMIAL = 137
BCH_BITS = 5
infoMessage(getLineNumber(), f'Starting to encode. \
Defined:   BCH_POLYNOMIAL = {BCH_POLYNOMIAL}    BCH_BITS = {BCH_BITS}')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=r'./images')
    parser.add_argument('--secret', type=str, default='Stega!!')
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    # pdb.set_trace()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return
    infoMessage(getLineNumber(), 'received input image')
    infoMessage(getLineNumber(), 'loading encoder')
    encoder = torch.load(args.model)
    encoder.eval()
    if args.cuda:
        encoder = encoder.cuda()
    infoMessage(getLineNumber(), 'encoder loaded')

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    infoMessage(getLineNumber(), f'beginning to use BCH library: bch={bch}')

    if len(args.secret) > 7:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    data = bytearray(args.secret + ' ' * (7 - len(args.secret)), 'utf-8')  # setting the length of secret to 7
    ecc = bch.encode(data)
    packet = data + ecc
    infoMessage(getLineNumber(), f'data = {data}  data length = {len(data)}')
    infoMessage(getLineNumber(), f'ecc = {ecc}')
    infoMessage(getLineNumber(), f'packet = {packet}')

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])
    infoMessage(getLineNumber(), f'secret = {secret}')
    infoMessage(getLineNumber(), f'packet binary = {packet_binary}')
    infoMessage(getLineNumber(), f'len((packet binary) = {len(packet_binary)}')
    secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0)
    if args.cuda:
        secret = secret.cuda()

    width = 400
    height = 400
    size = (width, height)
    to_tensor = transforms.ToTensor()

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        with torch.no_grad():
            for filename in files_list:
                infoMessage(getLineNumber(), 'convert photo to RGB, and convert to tensor')
                image = Image.open(filename).convert("RGB")
                image = ImageOps.fit(image, size)
                image = to_tensor(image).unsqueeze(0)
                if args.cuda:
                    image = image.cuda()

                residual = encoder((secret, image))
                encoded = image + residual
                if args.cuda:
                    residual = residual.cpu()
                    encoded = encoded.cpu()
                # todo: clip those values before casting to uint8
                encoded = np.array(encoded.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))  # todo: check if the values are not being trimmed here

                residual = residual[0] + .5
                residual = np.array(residual.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))  # todo: check if the values are not being trimmed here

                save_name = os.path.basename(filename).split('.')[0]

                infoMessage(getLineNumber(), 'saving encoded')
                im = Image.fromarray(encoded)
                im.save(args.save_dir + '/' + save_name + '_hidden.png')

                infoMessage(getLineNumber(), 'saving residual')
                im = Image.fromarray(residual)
                im.save(args.save_dir + '/' + save_name + '_residual.png')


if __name__ == "__main__":
    main()
