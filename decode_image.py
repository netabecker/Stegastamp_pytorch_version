from aux_functions import *
import glob
import bchlib
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    decoder = torch.load(args.model)
    decoder.eval()
    if args.cuda:
        decoder = decoder.cuda()

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    width = 400
    height = 400
    size = (width, height)
    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = ImageOps.fit(image, size)
            image = to_tensor(image).unsqueeze(0)
            if args.cuda:
                image = image.cuda()

            secret = decoder(image)

            if args.cuda:
                secret = secret.cpu()
            secret = np.array(secret[0])
            secret = np.round(secret)

            infoMessage(getLineNumber(), f'secret = {secret}')

            packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
            infoMessage(getLineNumber(), f'packet binary = {packet_binary}')
            packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
            packet = bytearray(packet)
            infoMessage(getLineNumber(), f'packet = {packet}')

            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

            infoMessage(getLineNumber(), f'bch.ecc_bytes = {bch.ecc_bytes}')
            infoMessage(getLineNumber(), f'data = {data}')
            infoMessage(getLineNumber(), f'len(data) = {len(data)}')
            infoMessage(getLineNumber(), f'ecc = {ecc}')
            infoMessage(getLineNumber(), f'len(ecc) = {len(ecc)}')

            bitflips = bch.decode_inplace(data, ecc)
            infoMessage(getLineNumber(), f'bitflips = {bitflips}')
            print(f'bitflips: {bitflips}')    # debug
            if bitflips != -1:
                try:
                    code = data.decode("utf-8")
                    print(f'This is the code: \n{code}\n')  # debug
                    print(filename, code)
                    continue
                except:
                    continue
            print(filename, 'Failed to decode')


if __name__ == "__main__":
    main()
