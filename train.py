from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss

import numpy as np
import torch
from skimage import morphology
import cv2
def skeleton_get(img, args):  # tensor

    re = []
    img = img.cpu()
    for i in range(img.size(0)):
        x = img[i, :, :, :]
        x = x.permute(1, 2, 0)
        x = x.numpy()
        # print(type(x))

        _, binary = cv2.threshold(x, 200/255, 255/255, cv2.THRESH_BINARY_INV)  # 二值化处理

        binary[binary == 255/255] = 1
        skeleton0 = morphology.skeletonize(binary)  # 骨架提取
        skeleton = skeleton0.astype(np.uint8) * 255

        gray = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)

        dst = 255/255 - gray

        stacked_img = np.stack((dst,) * 3, axis=-1)
        stacked_img = torch.tensor(stacked_img)
        stacked_img = stacked_img.permute(2, 0, 1)
        # print(stacked_img.shape)
        re.append(stacked_img)
        img_skeleton_tensor = torch.stack(re)
        img_skeleton_tensor =img_skeleton_tensor.cuda(args.gpu)

    return img_skeleton_tensor


def trainGAN(data_loader, networks, opts, epoch, args, additional):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_rec = AverageMeter()
    L1 = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D = networks['D']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module

    # set opts
    d_opt = opts['D']
    g_opt = opts['G']
    c_opt = opts['C']
    # switch to train mode
    D.train()
    G.train()
    C.train()
    C_EMA.train()
    G_EMA.train()

    logger = additional['logger']


    # summary writer
    #train_it = iter(data_loader)
    train_content = iter(data_loader['content'])  # 原始+标签
    train_style = iter(data_loader['style'])  # 目标+标签+原始
    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        try:
            imgs_content, content_label, Ground_Truth = next(train_content)
            imgs_style, style_label, _ = next(train_style)  #这个也得

        except:
            train_content = iter(data_loader['content'])
            train_style = iter(data_loader['style'])
            imgs_content, content_label, Ground_Truth = next(train_content)
            imgs_style, style_label, _ = next(train_style)

        imgs_content = imgs_content.cuda(args.gpu)#内容图片
        Ground_Truth = Ground_Truth.cuda(args.gpu)#真实（风格）图片
        #imgs_style = Ground_Truthone-->one
        imgs_style = imgs_style.cuda(args.gpu)#随机风格图片
        style_label = style_label.cuda(args.gpu)
        content_label = content_label.cuda(args.gpu)

        imgs_content_skeleton = skeleton_get(imgs_content,args)#获得内容图片的骨架
        Ground_Truth_skeleton = skeleton_get(Ground_Truth, args)#获得风格图片的骨架
        imgs_content_skeleton = imgs_content_skeleton.float()
        Ground_Truth_skeleton = Ground_Truth_skeleton.float()
        # x_ref_idx = torch.randperm(imgs_content.size(0))
        #

        # x_ref_idx = x_ref_idx.cuda(args.gpu)
        # x_ref = imgs_style[x_ref_idx]

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################
        with torch.no_grad():
            # y_ref = imgs_content.clone()
            # y_ref = y_ref[x_ref_idx]
            # s_ref = C.moco(imgs_style)
            # c_src, skip1, skip2 = G.cnt_encoder(imgs_content)
            # x_fake, _ = G.decode(c_src, s_ref, skip1, skip2)



            # s_ref = C.moco(imgs_style)
            s_ref = C.split(imgs_style)

            c_src, skip1, skip2 = G.cnt_encoder(imgs_content)
            content_skeleton = G.skeleton(imgs_content_skeleton)
            content_fusion_skeleton =G.fusion(c_src,content_skeleton)

            x_fake, _ = G.decode(content_fusion_skeleton, s_ref, skip1, skip2)

            #s_ref_cor = C.moco(Ground_Truth)#andom+order--->D
            #x_fake_corr,_ =G.decode(c_src,s_ref_cor,skip1,skip2)

        #x_ref.requires_grad_()
        imgs_style.requires_grad_()
        Ground_Truth.requires_grad_()

        # Train D
        d_real_logit, _ = D(Ground_Truth, style_label)
        d_fake_logit, _ = D(x_fake.detach() , style_label)

        #d_fake_logit_corr, _ = D(x_fake_corr.detach(), style_label)#x_fake_corr

        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        #d_adv_fake_corr = calc_adv_loss(d_fake_logit_corr, 'd_fake')
        d_adv = d_adv_real + d_adv_fake   #+ d_adv_fake_corr
#这个是内容还是风格
        d_gp = args.w_gp * compute_grad_gp(d_real_logit, Ground_Truth, is_patch=False)

        x_fake_skeleton = skeleton_get(x_fake,args)
        l1_skeleton = calc_recon_loss(x_fake_skeleton, Ground_Truth_skeleton)

        d_loss = d_adv + d_gp +l1_skeleton *0.01
        l1_skeleton .requires_grad_()

        d_opt.zero_grad()
        d_adv_real.backward(retain_graph=True)
        d_gp.backward()
        d_adv_fake.backward()
        l1_skeleton.backward()
        if args.distributed:
            average_gradients(D)
        d_opt.step()

        # Train G
        s_ref = C.split(imgs_style)
        #s_ref = C.moco(imgs_style)
        c_src, skip1, skip2 = G.cnt_encoder(imgs_content)
        content_skeleton = G.skeleton(imgs_content_skeleton)
        content_fusion_skeleton = G.fusion(c_src, content_skeleton)
        x_fake, _ = G.decode(content_fusion_skeleton, s_ref, skip1, skip2)
        s_src = C.moco(imgs_content)
        s_ref = C.moco(Ground_Truth)
        #s_ref = C.moco(x_ref)

        c_src, skip1, skip2 = G.cnt_encoder(Ground_Truth)
        Gt_skeleton = G.skeleton(Ground_Truth_skeleton)
        content_fusion_skeleton = G.fusion(c_src, Gt_skeleton)
        Gt_fusion_skeleton = G.fusion(c_src, Gt_skeleton)
        x_fake_ss, offset_loss = G.decode(content_fusion_skeleton, s_ref, skip1, skip2)#style+style-->style
        x_rec, _ = G.decode(Gt_fusion_skeleton, s_src, skip1, skip2)#style+content-->content

        g_fake_logit, _ = D(x_fake, style_label)#style+style
        g_rec_logit, _ = D(x_rec, content_label)#style+content

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        g_adv = g_adv_fake + g_adv_rec

        g_imgrec = calc_recon_loss(x_rec, imgs_content)

        c_x_fake, _, _ = G.cnt_encoder(x_fake_ss)#style=style-->style--->中间过程
        g_conrec = calc_recon_loss(c_x_fake, c_src)

        l1_re = calc_recon_loss(x_fake_ss,Ground_Truth)
        l1 =  calc_recon_loss(x_fake,Ground_Truth)
        l1 = l1_re +l1


        g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec +args.w_rec * g_conrec + args.w_off * offset_loss+l1*0.5
 
        g_opt.zero_grad()
        c_opt.zero_grad()
        g_loss.backward()
        if args.distributed:
            average_gradients(G)
            average_gradients(C)
        c_opt.step()
        g_opt.step()

        ##################
        # END Train GANs #
        ##################


        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)
        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), imgs_content.size(0))
                d_advs.update(d_adv.item(), imgs_content.size(0))
                d_gps.update(d_gp.item(), imgs_content.size(0))

                g_losses.update(g_loss.item(), imgs_content.size(0))
                g_advs.update(g_adv.item(), imgs_content.size(0))
                g_imgrecs.update(g_imgrec.item(), imgs_content.size(0))
                g_rec.update(g_conrec.item(), imgs_content.size(0))
                L1.update(l1.item(),imgs_content.size(0))

                moco_losses.update(offset_loss.item(), imgs_content.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/conrec', g_rec.avg, summary_step)
                add_logs(args, logger, 'C/OFFSET', moco_losses.avg, summary_step)
                add_logs(args,logger,'G/l1',L1.avg,summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(epoch + 1, args.epochs, i+1, args.iters,
                                                        training_mode, d_losses=d_losses, g_losses=g_losses))

    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)

