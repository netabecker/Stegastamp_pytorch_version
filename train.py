from aux_functions import *

infoMessage(getLineNumber(), 'Successfully imported aux lib')
from inspect import currentframe
import os
import yaml
import random
import model
# import model_by_tensor as model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
import utils
from dataset import StegaData
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lpips

with open('cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

if not os.path.exists(args.saved_models):
    os.makedirs(args.saved_models)


def main():
    infoMessage(getLineNumber(), 'Beginning of main')
    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)

    infoMessage(getLineNumber(), 'Loading the data')
    dataset = StegaData(args.train_path, args.secret_size, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    encoder = model.StegaStampEncoder()
    decoder = model.StegaStampDecoder(secret_size=args.secret_size)
    discriminator = model.Discriminator()
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    if args.cuda:
        infoMessage(getLineNumber(), 'Cuda = True')
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()
        lpips_alex.cuda()

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    height = 400
    width = 400

    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0

    infoMessage(getLineNumber(), 'Running over the data')
    infoMessage(getLineNumber(), 'initializing loss arrays')

    """ 
    loss_array = []
    # lpips_loss_array = []
    secret_loss_array = []
    D_loss_array = []
    """

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            image_input, secret_input = next(iter(dataloader))
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
            no_im_loss = global_step < args.no_im_loss_steps
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            # lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)
            # G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
            # l2_edge_gain = 0
            # if global_step > args.l2_edge_delay:
            #     l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp,
            #                        args.l2_edge_gain)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1
            Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
            if args.cuda:
                Ms = Ms.cuda()

            """ New version - after removing all loss scales"""

            loss_scales = [l2_loss_scale, 0, secret_loss_scale, 0]  # setting the weight to 0 of unknown losses
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(encoder, decoder, discriminator, lpips_alex,
                                                                            secret_input, image_input,
                                                                            args.l2_edge_gain, args.borders,
                                                                            args.secret_size, Ms, loss_scales,
                                                                            yuv_scales, args, global_step, writer)

            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()

            # ------> PREVIOUS VERSION - BEFORE REMOVING ALL DIFFERENT LOSSES
            """
            loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(encoder, decoder, discriminator, lpips_alex,
                                                                            secret_input, image_input,
                                                                            args.l2_edge_gain, args.borders,
                                                                            args.secret_size, Ms, loss_scales,
                                                                            yuv_scales, args, global_step, writer)
            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()
            """

            print('{:g}: Loss = {:.4f}'.format(global_step, loss))

            writer.add_scalars('Loss values', {'loss': loss.item(), 'secret loss': secret_loss.item(),
                                               'D_loss loss': D_loss.item()})

            """ 
            ---> Create graphs localy
            
            loss_array.append(loss.item())
            secret_loss_array.append(secret_loss.item())
            D_loss_array.append(D_loss.item())

            if (global_step > 1000 and global_step % 5000 == 0) or (global_step < 5000 and global_step % 1000 == 0):
                infoMessage(getLineNumber(), f'loss type: {loss.type()}')
                graph_create(global_step, loss_array, secret_loss_array, D_loss_array, graph_labels_created)
            """
    writer.close()
    torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
    torch.save(decoder, os.path.join(args.saved_models, "decoder.pth"))


if __name__ == '__main__':
    main()
