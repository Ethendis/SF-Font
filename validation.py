import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import torch.nn.functional as F

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from scipy import linalg

from tools.utils import *

from train import skeleton_get
def validateUN(data_loader, networks, epoch, args, additional=None):
    # set nets
    D = networks['D']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    # switch to train mode

    D.eval()
    G.eval()
    C.eval()
    C_EMA.eval()
    G_EMA.eval()
    # data loader
    val_dataset = data_loader['TRAINSET']#得到每个数据集后50个字作为测试集
    # val_loader = data_loader['VAL']

    #生成时用生成个数
    #train_num = 1500
    train_num = 5091
    #训练时用50
    #train_num = 50
    x_each_cls = []
    x_each_cls_train = []
    path = os.path.join(args.res_dir, '{}'.format(epoch + 1))
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            os.makedirs(os.path.join(path, 'train'))
            os.makedirs(os.path.join(path, 'test'))
        except OSError:
            print("you are repeating makedirs!")

    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):
            # tmp_cls_set = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()[-args.val_num:]
            tmp_cls_set = torch.nonzero(val_tot_tars == args.att_to_use[cls_idx], as_tuple=False)[-args.val_num:]#得到序号
            tmp_ds = torch.utils.data.Subset(val_dataset, tmp_cls_set)
            tmp_dl = torch.utils.data.DataLoader(tmp_ds, batch_size=args.val_num, shuffle=False,
                                                 num_workers=0, pin_memory=True, drop_last=False)
            tmp_iter = iter(tmp_dl)#加载数据
            tmp_sample = None
            for sample_idx in range(len(tmp_iter)):
                imgs, _ ,_= next(tmp_iter)
                x_ = imgs
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)
            x_each_cls.append(tmp_sample)#测试集中的每一个汉字



    with torch.no_grad():
        val_tot_tars = torch.tensor(val_dataset.targets)
        for cls_idx in range(len(args.att_to_use)):
            # tmp_cls_set = (val_tot_tars == args.att_to_use[cls_idx]).nonzero()[-args.val_num:]
            tmp_cls_set_train = torch.nonzero(val_tot_tars == args.att_to_use[cls_idx], as_tuple=False)[:train_num]
            tmp_ds_train = torch.utils.data.Subset(val_dataset, tmp_cls_set_train)
            tmp_dl_train = torch.utils.data.DataLoader(tmp_ds_train, batch_size=args.val_num, shuffle=False,
                                                 num_workers=0, pin_memory=True, drop_last=False)
            tmp_iter_train = iter(tmp_dl_train)
            tmp_sample = None
            for sample_idx in range(len(tmp_iter_train)):
                imgs, _,_ = next(tmp_iter_train)
                x_ = imgs
                if tmp_sample is None:
                    tmp_sample = x_.clone()
                else:
                    tmp_sample = torch.cat((tmp_sample, x_), 0)
            x_each_cls_train.append(tmp_sample)#测试集中的前train_num个汉字
    
    if epoch >= args.fid_start-1:
        # Reference guided
        with torch.no_grad():#测试训练集
            # Just a buffer image ( to make a grid )

            for class_index_sty in range(len(args.att_to_use)):  # 生成的风格图像目标

                sty_fonts = x_each_cls_train[class_index_sty][:train_num, :, :, :].cuda(args.gpu, non_blocking=True)
                for class_index_con in range(len(args.att_to_use)):
                    content_fonts = x_each_cls_train[class_index_con][:train_num, :, :, :].cuda(args.gpu,
                                                                                                non_blocking=True)
                    if class_index_sty <= class_index_con:
                        pass
                    else:

                        x_ref_idx = torch.randperm(content_fonts.size(0))
                        # sty_fonts_randoms = sty_fonts[x_ref_idx]
                        for sample_idx in range(0, train_num // args.val_batch, 1):
                            index = sample_idx * args.val_batch
                            sty_font = sty_fonts[index: index + args.val_batch]
                            content_font = content_fonts[index: index + args.val_batch]
                            # sty_fonts_random = sty_fonts_randoms[index: index + args.val_batch]
                            c_src, skip1, skip2 = G_EMA.cnt_encoder(content_font)  # 生成内容向量

                            content_font_sktleton = skeleton_get(content_font,args)

                            content_font_sktleton = content_font_sktleton.float()

                            content_skeleton_feature  = G.skeleton(content_font_sktleton)
                            import matplotlib.pyplot as plt


                            fusion = G_EMA.fusion(c_src,content_skeleton_feature)



                            s_ref = C_EMA(sty_font, sty=True)  # 生成风格向量#因为有true，所有返回的即是128维向量

                            # c_src  = c_src.mul(s_ref)
                            G_image, _ = G_EMA.decode(fusion, s_ref, skip1, skip2)  # 相同字体内容情况下的生成结果
                            # exit()
                            result = G_image
                            # result = torch.cat((content_font, G_image), 0)
                            #result = torch.cat((result, sty_font), 0)

                            vutils.save_image(result, os.path.join(args.res_dir,
                                                                   '{}/train/train_{}_{}_{}_{}.png'.format(
                                                                       epoch + 1, epoch + 1, class_index_sty,
                                                                       class_index_con, sample_idx)),
                                              normalize=True, nrow=3)


                            # vutils.save_image(G_image, os.path.join(args.res_dir,
                            #                                        '{}/train/{}.jpg'.format(
                            #                                            epoch + 1, sample_idx)),
                            #                   normalize=True, nrow=1)

        with torch.no_grad():
            for class_index_sty in range(len(args.att_to_use)):  # 生成的风格图像目标
                sty_fonts = x_each_cls[class_index_sty][:train_num, :, :, :].cuda(args.gpu, non_blocking=True)
                for class_index_con in range(len(args.att_to_use)):
                    content_fonts = x_each_cls[class_index_con][:args.val_num, :, :, :].cuda(args.gpu,
                                                                                             non_blocking=True)
                    if class_index_sty <= class_index_con:
                        pass
                    else:
                        x_ref_idx = torch.randperm(content_fonts.size(0))
                        sty_fonts_randoms = sty_fonts[x_ref_idx]
                        for sample_idx in range(0, args.val_num // args.val_batch, 1):
                            index = sample_idx * args.val_batch
                            sty_font = sty_fonts[index: index + args.val_batch]
                            content_font = content_fonts[index: index + args.val_batch]
                            sty_fonts_random = sty_fonts_randoms[index: index + args.val_batch]
                            c_src, skip1, skip2 = G_EMA.cnt_encoder(content_font)  # 生成内容向量
                            content_font_sktleton = skeleton_get(content_font,args)
                            content_font_sktleton= content_font_sktleton.float()
                            content_skeleton_feature = G.skeleton(content_font_sktleton)
                            fusion = G_EMA.fusion(c_src, content_skeleton_feature)
                            s_ref = C_EMA(sty_fonts_random, sty=True)  # 生成风格向量#因为有true，所有返回的即是128维向量

                            # c_src  = c_src.mul(s_ref)
                            G_image, _ = G_EMA.decode(fusion, s_ref, skip1, skip2)  # 相同字体内容情况下的生成结果
                            result = G_image
                            # result = torch.cat((content_font, G_image), 0)
                            # result = torch.cat((result, sty_font), 0)
                            vutils.save_image(result, os.path.join(args.res_dir,
                                                                   '{}/test/Test_{}_{}_{}_{}.png'.format(epoch + 1,
                                                                                                         epoch + 1,
                                                                                                         class_index_sty,
                                                                                                         class_index_con,
                                                                                                         sample_idx)),
                                              normalize=True, nrow=3)

