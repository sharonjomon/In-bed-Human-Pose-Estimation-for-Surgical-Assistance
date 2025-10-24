'''
2d pose estimation handling
'''

from data.SLP_RD import SLP_RD
from data.SLP_FD import SLP_FD
import utils.vis as vis
import utils.utils as ut
import numpy as np
import opt
import cv2
import torch
import json
import shutil
from os import path
import os
from utils.logger import Colorlogger
from utils.utils_tch import get_model_summary
from core.loss import JointsMSELoss
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
from utils.utils_ds import accuracy, flip_back
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt

# opts outside?
opts = opt.parseArgs()
if 'depth' in opts.mod_src[0]:  # the leading modalities, only depth use tight bb other raw image size
    opts.if_bb = True  # not using bb, give ori directly
else:
    opts.if_bb = False  #
exec('from model.{} import get_pose_net'.format(opts.model))  # pose net in
opts = opt.aug_opts(opts)  # add necesary opts parameters   to it
# opt.print_options(opts)



def train(loader, ds_rd, model, criterion, optimizer, epoch, phase, n_iter=-1, logger=None, opts=None, visualizer=None, device="cpu"):
    '''
    iter through epoch , return rst{'acc', loss'} each as list can be used outside for updating.
    :param loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:  for print infor
    :param n_iter: the iteration wanted, -1 for all iters
    :param opts: keep some additional controls
    :param visualizer: for visualizer
    :return:
    '''
    batch_time = ut.AverageMeter()
    data_time = ut.AverageMeter()
    losses = ut.AverageMeter()
    acc = ut.AverageMeter()
    acc_right = ut.AverageMeter()
    acc_left = ut.AverageMeter()
    figures_dir = os.path.join(opts.model_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # switch to train mode
    model.train()
    end = time.time()
    li_loss = []
    li_acc = []
    for i, inp_dct in enumerate(loader):
        # get items
        if i>=n_iter and n_iter>0:    # break if iter is set and i is greater than that
            break
        input = inp_dct['pch'].to(device, non_blocking=True)
        target = inp_dct['hms'].to(device, non_blocking=True)     # 14 x 64 x 1??
        target_weight = inp_dct['joints_vis'].to(device, non_blocking=True)

        # measure data loading time     weight, visible or not
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)      # no need to cuda it?

        target = target.to(device, non_blocking=True)
        target_weight = target_weight.to(device, non_blocking=True)

        if hasattr(model, "compute_loss"):
            losses_dict = model.compute_loss(outputs, target, target_weight, criterion, phase=opts.phase)
            loss = losses_dict['total_loss']
            loss_right = losses_dict['loss_right']
            loss_left = losses_dict['loss_left']
            print(f"Loss Right Wrist: {loss_right.item():.5f}, Loss Left Wrist: {loss_left.item():.5f}")
        else:
            # Fallback for Phase 1 (StackedHourGlass)
            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                loss = criterion(outputs, target, target_weight)

        if phase in ["2a", "2b"] and isinstance(outputs, dict):
            refined = outputs['refined_wrists']

            # Visualize refined right wrist heatmap (first sample)
            if epoch % 20 == 0 and i == 0:  # only show once to avoid flooding
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()

                plt.imshow(refined['right'][0, 0].detach().cpu().numpy(), cmap='hot')
                plt.title("Refined Right Wrist Heatmap")
                plt.savefig(os.path.join(figures_dir, f"refined_right_wrist_epoch{epoch}_iter{i}.png"))
                plt.close()

                right_forearm = outputs['right_forearm']
                #left_forearm = outputs['left_forearm']
                heatmaps = outputs['heatmaps']
                right_roi = outputs['right_roi']
                left_roi = outputs['left_roi']
                wrist_coords = model.wrist_refine.soft_argmax_2d(refined['right'])  # shape: (B, 2)
                wrist_x, wrist_y = wrist_coords[0].detach().cpu().numpy()

                ax.imshow(right_forearm[0].squeeze(0).cpu().numpy(), cmap='gray')
                ax.scatter(wrist_x, wrist_y, color='red')


                ax.set_title("Right Forearm Mask with Wrist Overlay")
                fig.savefig(os.path.join(figures_dir, f"refined_forearm_mask_epoch{epoch}_iter{i}.png"))
                plt.close()

                plt.imshow(heatmaps[-1][0, 7].detach().cpu().numpy(), cmap='hot')  # Right elbow
                plt.imshow(heatmaps[-1][0, 6].detach().cpu().numpy(), cmap='hot')  # Right wrist

                cropped_patch_right = right_roi[0, 0].detach().cpu().numpy()  # First sample, first channel
                plt.imshow(cropped_patch_right, cmap='hot')
                plt.title("Cropped Right ROI")
                plt.savefig(os.path.join(figures_dir, f"cropped_right_roi_epoch{epoch}_iter{i}.png"))
                plt.close()

                cropped_patch_left = left_roi[0, 0].detach().cpu().numpy()  # First sample, first channel
                plt.imshow(cropped_patch_left, cmap='hot')
                plt.title("Cropped Left ROI")
                plt.savefig(os.path.join(figures_dir, f"cropped_left_roi_epoch{epoch}_iter{i}.png"))
                plt.close()




            # Right wrist
            right_pred = refined['right'].detach().cpu().numpy()
            right_gt = target[:, 6:7].detach().cpu().numpy()
            _, right_acc, _, _ = accuracy(right_pred, right_gt)
            acc_right.update(right_acc, input.size(0))

            # Left wrist
            left_pred = refined['left'].detach().cpu().numpy()
            left_gt = target[:, 11:12].detach().cpu().numpy()
            _, left_acc, _, _ = accuracy(left_pred, left_gt)
            acc_left.update(left_acc, input.size(0))


        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Extract final heatmap output for accuracy computation
        if isinstance(outputs, dict):
            output = outputs['heatmaps'][-1] if isinstance(outputs['heatmaps'], list) else outputs['heatmaps']
        else:
            output = outputs[-1] if isinstance(outputs, list) else outputs

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())  # hm directly, with normalize with 1/10 dim,  pck0.5,  cnt: n_smp,  pred
        acc.update(avg_acc, cnt)  # keep average acc

        if visualizer and 0 == i % opts.update_html_freq:     # update current result, get vis dict
            n_jt = ds_rd.joint_num_ori
            mod0 = opts.mod_src[0]
            mean = ds_rd.means[mod0]
            std = ds_rd.stds[mod0]
            img_patch_vis = ut.ts2cv2(input[0], mean, std)  # to CV BGR, mean std control channel detach inside
            # pseudo change
            # Apply colormap only if defined for this modality
            cm_name = ds_rd.dct_clrMap.get(mod0, None)
            if cm_name is not None and isinstance(cm_name, str):  # not None or empty
                cm = getattr(cv2, cm_name)
                img_patch_vis = cv2.applyColorMap(img_patch_vis, cm)[..., ::-1]  # convert BGR->RGB
            # else: leave img_patch_vis as is for RGB

            # get pred
            pred2d_patch = np.ones((n_jt, 3))  # 3rd for  vis
            pred2d_patch[:, :2] = pred[0] / opts.out_shp[0] * opts.sz_pch[1]
            img_skel = vis.vis_keypoints(img_patch_vis, pred2d_patch, ds_rd.skels_idx)

            hm_gt = target[0].cpu().detach().cpu().numpy().sum(axis=0)    # HXW
            hm_gt = ut.normImg(hm_gt)

            hm_pred = output[0].detach().cpu().numpy().sum(axis=0)
            hm_pred = ut.normImg(hm_pred)
            img_cb = vis.hconcat_resize([img_skel, hm_gt, hm_pred])
            vis_dict = {'img_cb': img_cb}
            visualizer.display_current_results(vis_dict, epoch, False)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opts.print_freq == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            li_loss.append(losses.val)   # the current loss
            li_acc.append(acc.val)

    return {'losses':li_loss, 'accs':li_acc, 'acc_right': acc_right.avg, 'acc_left': acc_left.avg}


def validate(loader, ds_rd, model, criterion, phase, n_iter=-1, logger=None, opts=None, if_svVis=False, visualizer=None, device="cpu"):
    '''
    loop through loder, all res, get preds and gts and normled dist.
    With flip test for higher acc.
    for preds, bbs, jts_ori, jts_weigth out, recover preds_ori, dists_nmd, pckh( dist and joints_vis filter, , print, if_sv then save all these
    :param loader:
    :param ds_rd: the reader, givens the length and flip pairs
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param n_iter:
    :param logger:
    :param opts:
    :return:
    '''
    batch_time = ut.AverageMeter()
    losses = ut.AverageMeter()
    acc = ut.AverageMeter()
    acc_right = ut.AverageMeter()
    acc_left = ut.AverageMeter()


    # switch to evaluate mode
    model.eval()

    num_samples = ds_rd.n_smpl
    n_jt = ds_rd.joint_num_ori

    # to accum rst
    preds_hm = []
    bbs = []
    li_joints_ori = []
    li_joints_vis = []
    li_l_std_ori = []
    with torch.no_grad():
        end = time.time()
        for i, inp_dct in enumerate(loader):
            # compute output
            input = inp_dct['pch'].to(device, non_blocking=True)
            target = inp_dct['hms'].to(device, non_blocking=True)
            target_weight = inp_dct['joints_vis'].to(device, non_blocking=True)
            bb = inp_dct['bb'].to(device, non_blocking=True)
            joints_ori = inp_dct['joints_ori'].to(device, non_blocking=True)
            l_std_ori = inp_dct['l_std_ori'].to(device, non_blocking=True)
            if i>= n_iter and n_iter>0:     # limiting iters
                break
            outputs = model(input)

            # Extract final heatmap tensor from outputs
            if isinstance(outputs, dict):
                 output = outputs['heatmaps'][-1] if isinstance(outputs['heatmaps'], list) else outputs['heatmaps']
            else:
                 output = outputs[-1] if isinstance(outputs, list) else outputs
            output_ori = output.clone()
            if opts.if_flipTest:
                input_flipped = input.flip(3).clone()       # flipped input
                outputs_flipped = model(input_flipped)      # flipped output
                if isinstance(outputs_flipped, dict):
                    output_flipped = outputs_flipped['heatmaps'][-1] if isinstance(outputs_flipped['heatmaps'], list) else outputs_flipped['heatmaps']
                else:
                    output_flipped = outputs_flipped[-1] if isinstance(outputs_flipped, list) else outputs_flipped

                output_flipped_ori = output_flipped.clone()
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           ds_rd.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).to(device) # N x n_jt xh x w tch

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if_shiftHM = True  # no idea why
                if if_shiftHM:      # check original
                    # print('run shift flip')
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.to(device, non_blocking=True)
            target_weight = target_weight.to(device, non_blocking=True)
            if hasattr(model, "compute_loss"):
                loss = model.compute_loss(outputs, target, target_weight, criterion, phase)
            else:
                # Fallback for Phase 1 (StackedHourGlass)
                if isinstance(outputs, list):
                    loss = criterion(outputs[0], target, target_weight)
                    for output in outputs[1:]:
                        loss += criterion(output, target, target_weight)
                else:
                    loss = criterion(outputs, target, target_weight)

            if phase in ["2a", "2b"] and isinstance(outputs, dict):
                refined = outputs['refined_wrists']

                # Right wrist
                right_pred = refined['right'].detach().cpu().numpy()
                right_gt = target[:, 6:7].detach().cpu().numpy()
                _, right_acc, _, _ = accuracy(right_pred, right_gt)
                acc_right.update(right_acc, input.size(0))

                # Left wrist
                left_pred = refined['left'].detach().cpu().numpy()
                left_gt = target[:, 11:12].detach().cpu().numpy()
                _, left_acc, _, _ = accuracy(left_pred, left_gt)
                acc_left.update(left_acc, input.size(0))

            num_images = input.size(0)
            # measure accuracy and record loss
            if isinstance(loss, dict):
                losses.update(loss['total_loss'].item(), num_images)
            else:
                losses.update(loss.item(), num_images)

            _, avg_acc, cnt, pred_hm = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            acc.update(avg_acc, cnt)

            # preds can be furhter refined with subpixel trick, but it is already good enough.
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # keep rst
            preds_hm.append(pred_hm)        # already numpy, 2D
            bbs.append(bb.cpu().numpy())
            li_joints_ori.append(joints_ori.cpu().numpy())
            li_joints_vis.append(target_weight.cpu().numpy())
            li_l_std_ori.append(l_std_ori.cpu().numpy())

            if if_svVis and 0 == i % opts.svVis_step:
                sv_dir = opts.vis_test_dir  # exp/vis/Human36M
                # batch version
                mod0 = opts.mod_src[0]
                mean = ds_rd.means[mod0]
                std = ds_rd.stds[mod0]
                img_patch_vis = ut.ts2cv2(input[0], mean, std) # to CV BGR
                img_patch_vis_flipped = ut.ts2cv2(input_flipped[0], mean, std) # to CV BGR
                # pseudo change
                cm_name = ds_rd.dct_clrMap.get(mod0, None) 
                if isinstance(cm_name, str):
                    cm = getattr(cv2, cm_name)
                    img_patch_vis = cv2.applyColorMap(img_patch_vis, cm)
                    img_patch_vis_flipped = cv2.applyColorMap(img_patch_vis_flipped, cm)

                # original version get img from the ds_rd , different size , plot ing will vary from each other
                # warp preds to ori
                # draw and save  with index.

                idx_test = i * opts.batch_size  # image index
                skels_idx = ds_rd.skels_idx
                # get pred2d_patch
                pred2d_patch = np.ones((n_jt, 3))  # 3rd for  vis
                pred2d_patch[:,:2] = pred_hm[0] / opts.out_shp[0] * opts.sz_pch[1]      # only first
                vis.save_2d_skels(img_patch_vis, pred2d_patch, skels_idx, sv_dir, suffix='-'+mod0,
                                  idx=idx_test)  # make sub dir if needed, recover to test set index by indexing.
                # Refined wrist visualization
                refined_right = outputs['refined_wrists']['right']  # (B, 1, H, W)
                refined_left  = outputs['refined_wrists']['left']

                # Convert to coordinates
                refined_right_coords = model.wrist_refine.soft_argmax_2d(refined_right)  # (B, 2)
                refined_left_coords  = model.wrist_refine.soft_argmax_2d(refined_left)

                # Visualize on same image
                img_patch_vis_rgb = img_patch_vis[..., ::-1]  # Convert BGR to RGB if needed
                fig, ax = plt.subplots()
                ax.imshow(img_patch_vis_rgb.astype(np.uint8))
                
                # Backbone elbow coordinates
                x_elbow_r, y_elbow_r = pred_hm[0][7] / opts.out_shp[0] * opts.sz_pch[1]  # Right elbow
                x_elbow_l, y_elbow_l = pred_hm[0][10] / opts.out_shp[0] * opts.sz_pch[1]  # Left elbow

                # Refined wrist coordinates
                x_ref_r, y_ref_r = refined_right_coords[0].detach().cpu().numpy()
                x_ref_r = x_ref_r / opts.out_shp[0] * opts.sz_pch[1]
                y_ref_r = y_ref_r / opts.out_shp[1] * opts.sz_pch[0]

                x_ref_l, y_ref_l = refined_left_coords[0].detach().cpu().numpy()
                x_ref_l = x_ref_l / opts.out_shp[0] * opts.sz_pch[1]
                y_ref_l = y_ref_l / opts.out_shp[1] * opts.sz_pch[0]


                # Draw refined elbow-to-wrist lines
                ax.plot([x_elbow_r, x_ref_r], [y_elbow_r, y_ref_r], color='red', linewidth=2, label='Refined Right Forearm')
                ax.plot([x_elbow_l, x_ref_l], [y_elbow_l, y_ref_l], color='red', linewidth=2, label='Refined Left Forearm')

                # Optional: draw refined wrist points
                ax.scatter(x_ref_r, y_ref_r, color='red')
                ax.scatter(x_ref_l, y_ref_l, color='red')

                # Optional: draw backbone wrist points for comparison
                x_back_r, y_back_r = pred_hm[0][6] / opts.out_shp[0] * opts.sz_pch[1]
                x_back_l, y_back_l = pred_hm[0][11] / opts.out_shp[0] * opts.sz_pch[1]
                ax.scatter(x_back_r, y_back_r, color='blue', label='Backbone R')
                ax.scatter(x_back_l, y_back_l, color='blue', label='Backbone L')

                ax.legend()
                fig.savefig(os.path.join(sv_dir, f"wrist_compare_overlay_{idx_test}.png"))
                plt.close(fig)                
                
                # save the hm images. save flip test
                hm_ori = ut.normImg(output_ori[0].cpu().numpy().sum(axis=0))    # rgb one
                hm_flip = ut.normImg(output_flipped[0].cpu().numpy().sum(axis=0))
                hm_flip_ori = ut.normImg(output_flipped_ori[0].cpu().numpy().sum(axis=0))
                # subFd = mod0+'_hmFlip_ori'
                # vis.save_img(hm_flip_ori, sv_dir, idx_test, sub=subFd)

                # combined
                # img_cb = vis.hconcat_resize([img_patch_vis, hm_ori, img_patch_vis_flipped, hm_flip_ori])        # flipped hm
                # subFd = mod0+'_cbFlip'
                # vis.save_img(img_cb, sv_dir, idx_test, sub=subFd)


            if i % opts.print_freq == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)

    preds_hm = np.concatenate(preds_hm,axis=0)      # N x n_jt  x 2
    bbs = np.concatenate(bbs, axis=0)
    joints_ori = np.concatenate(li_joints_ori, axis=0)
    joints_vis = np.concatenate(li_joints_vis, axis=0)
    l_std_ori_all = np.concatenate(li_l_std_ori, axis=0)

    preds_ori = ut.warp_coord_to_original(preds_hm, bbs, sz_out=opts.out_shp)
    err_nmd = ut.distNorm(preds_ori,  joints_ori, l_std_ori_all)
    ticks = np.linspace(0,0.5,11)   # 11 ticks
    pck_all = ut.pck(err_nmd, joints_vis, ticks=ticks)

    # save to plain format for easy processing
    rst = {
        'preds_ori':preds_ori.tolist(),
        'joints_ori':joints_ori.tolist(),
        'l_std_ori_all': l_std_ori_all.tolist(),
        'err_nmd': err_nmd.tolist(),
        'pck': pck_all.tolist(),
        'acc_right': acc_right.avg,
        'acc_left': acc_left.avg,
    }

    return rst
# -------------------------
# Helper functions
# -------------------------
def save_checkpoint(state, is_best, phase, model_dir, logger):
    checkpoint_file = os.path.join(model_dir, f'checkpoint_phase{phase}.pth')
    best_file = os.path.join(model_dir, f'model_best_phase{phase}.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)
        logger.info(f"Saved best model for phase {phase} at {best_file}")
    else:
        logger.info(f"Saved checkpoint for phase {phase} at {checkpoint_file}")

def _strip_module_prefix(sd):
    new_sd = {}
    for k, v in sd.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_sd[new_key] = v
    return new_sd

def load_state_dict_flexible(model, state_dict):
    """
    Try to load state_dict into model. If keys mismatch due to "module." prefix,
    try a prefix-stripped version, or the opposite.
    """
    try:
        model.load_state_dict(state_dict)
        return
    except Exception:
        # try strip 'module.' prefix
        try:
            stripped = _strip_module_prefix(state_dict)
            model.load_state_dict(stripped)
            return
        except Exception:
            # try adding 'module.' prefix (if model is DataParallel but state doesn't have module.)
            from collections import OrderedDict
            add_module = OrderedDict()
            for k, v in state_dict.items():
                add_module['module.' + k] = v
            model.load_state_dict(add_module)
            return

def load_checkpoint_for_phase(phase, model, optimizer, model_dir, device, load_optimizer_if_present=True):
    """
    Tries to load checkpoint for a given phase. Returns (found, start_epoch, best_acc)
    If optimizer is passed and load_optimizer_if_present True, loads optimizer state as well.
    """
    ckpt_path = os.path.join(model_dir, f'checkpoint_phase{phase}.pth')
    if not os.path.exists(ckpt_path):
        return False, 0, 0.0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # model state might be saved under 'state_dict'
    sd = ckpt.get('state_dict', ckpt)
    # load into model (flexible)
    load_state_dict_flexible(model, sd)
    start_epoch = ckpt.get('epoch', 0)
    best_acc = ckpt.get('best_acc', 0.0)
    if load_optimizer_if_present and optimizer is not None and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            # make sure optimizer tensors are on correct device (PyTorch usually handles this)
        except Exception as e:
            # sometimes optimizer state contains tensors on CPU; allow training with new optimizer
            print(f"Warning: couldn't fully load optimizer state for phase {phase}: {e}")
    return True, start_epoch, best_acc

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get logger
    if_test = opts.if_test
    if if_test:
        log_suffix = 'test'
    else:
        log_suffix = 'train'
    logger = Colorlogger(opts.log_dir, '{}_logs.txt'.format(log_suffix))    # avoid overwritting, will append
    opt.set_env(opts)
    opt.print_options(opts, if_sv=True)
    n_jt = SLP_RD.joint_num_ori     #

    # get model
    model = get_pose_net(in_ch=opts.input_nc, out_ch=n_jt)      # why call it get c
    dump_input = torch.rand((1, opts.input_nc, opts.sz_pch[1], opts.sz_pch[0]))
    logger.info(get_model_summary(model, dump_input))
    # Wrap with DataParallel if multiple GPUs (do this BEFORE loading checkpoints)
    if torch.cuda.is_available() and len(opts.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opts.gpu_ids)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(      # try to not use weights
        use_target_weight=True
    ).to(device)

    # ds adaptor
    SLP_rd_train = SLP_RD(opts, phase='train')  # all test result
    SLP_fd_train = SLP_FD(SLP_rd_train, opts, phase='train', if_sq_bb=True)
    train_loader = DataLoader(dataset=SLP_fd_train, batch_size= opts.batch_size // len(opts.trainset),
                        shuffle=True, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)

    SLP_rd_test = SLP_RD(opts, phase=opts.test_par)  # all test result      # can test against all controled in opt
    SLP_fd_test = SLP_FD(SLP_rd_test,  opts, phase='test', if_sq_bb=True)
    test_loader = DataLoader(dataset=SLP_fd_test, batch_size = opts.batch_size // len(opts.trainset),
                              shuffle=False, num_workers=opts.n_thread, pin_memory=opts.if_pinMem)

    # for visualzier
    if opts.display_id > 0:
        visualizer = Visualizer(opts)  # only plot losses here, a loss log comes with it,
    else:
        visualizer = None

    # ---------- BEGIN REPLACEMENT PHASE BLOCKS ----------
    # Make sure get_pose_net is imported/available in this file scope.

    # ---------------------------
    # Phase 1: Backbone Only (PLAIN Hourglass)
    # ---------------------------
    phase = "1"
    opts.phase = phase
    # Build plain hourglass (no wrist refine active in forward)
    model = get_pose_net(in_ch=opts.input_nc, out_ch=SLP_rd_train.joint_num_ori, refined=False)
    model = model.to(device)
    if torch.cuda.device_count() > 1 and opts.num_gpus > 1:
        model = torch.nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=opts.lr)
    found1, start_epoch1, best_acc1 = load_checkpoint_for_phase(phase, model, optimizer, opts.model_dir, device, load_optimizer_if_present=True)
    if found1:
        logger.info(f"Resuming Phase {phase} from epoch {start_epoch1}, best_acc={best_acc1:.4f}")
    else:
        start_epoch1 = getattr(opts, 'start_epoch', 0)
        if start_epoch1 < 0:
            start_epoch1 = 0
        best_acc1 = 0.0
        logger.info("No Phase 1 checkpoint found. Training Phase 1 from scratch.")

    phase1_done = False
    for epoch in range(start_epoch1, opts.end_epoch):
        rst_trn = train(train_loader, SLP_rd_train, model, criterion, optimizer, epoch,
                        n_iter=opts.trainIter, logger=logger, opts=opts, visualizer=visualizer, device=device, phase=phase)
        rst_test = validate(test_loader, SLP_rd_test, model, criterion,
                            n_iter=opts.trainIter, logger=logger, opts=opts, device=device, phase=phase)
        pck_all = rst_test['pck']

        # safe elbow idx
        try:
            right_elbow_idx = SLP_rd_test.joints_name.index('R_Elbow')
        except ValueError:
            right_elbow_idx = 7
        try:
            left_elbow_idx = SLP_rd_test.joints_name.index('L_Elbow')
        except ValueError:
            left_elbow_idx = 10

        right_acc = pck_all[right_elbow_idx][-1]
        left_acc = pck_all[left_elbow_idx][-1]

        is_best = max(right_acc, left_acc) > best_acc1
        best_acc1 = max(best_acc1, right_acc, left_acc)

        logger.info(f"Phase 1 Epoch {epoch}: Right Elbow PCK={right_acc:.3f}, Left Elbow PCK={left_acc:.3f}")

        titles = list(SLP_rd_test.joints_name[:SLP_rd_test.joint_num_ori]) + ['total']
        pckh05 = np.array(pck_all)[:, -1]
        ut.prt_rst([pckh05], titles, ['pckh0.5'], fn_prt=logger.info)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            'best_acc': best_acc1,
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, phase, opts.model_dir, logger)

        # criterion to move early
        if right_acc >= 85.0 and left_acc >= 85.0:
            logger.info("Both elbows reached 85% accuracy. Moving to Phase 2a.")
            phase1_done = True
            break

    if not phase1_done:
        logger.info("Phase 1 finished all configured epochs without hitting 85%. Proceeding to Phase 2a anyway.")

    # Ensure we have the final Phase1 checkpoint file path (used to initialize Phase2a)
    phase1_ckpt = os.path.join(opts.model_dir, f'checkpoint_phase1.pth')
    # Note: save_checkpoint already wrote checkpoint_phase1.pth in save_checkpoint( ... , phase = "1" ) earlier.


    # ---------------------------
    # Phase 2a: Warm-up (frozen backbone, train wrist refine only)
    # ---------------------------
    phase = "2a"
    opts.phase = phase
    ck2a_path = os.path.join(opts.model_dir, f'checkpoint_phase{phase}.pth')

    # Create refined model instance (with wrist refinement enabled)
    # We instantiate a new model object (refined=True) and then load backbone weights from Phase1
    refined_model = get_pose_net(in_ch=opts.input_nc, out_ch=SLP_rd_train.joint_num_ori, refined=True)
    refined_model = refined_model.to(device)
    if torch.cuda.device_count() > 1 and opts.num_gpus > 1:
        refined_model = torch.nn.DataParallel(refined_model)

    # If there is a Phase2a checkpoint, resume normally into refined_model
    if os.path.exists(ck2a_path):
        # We need an optimizer built from params that will be trained in Phase2a (refine only)
        # First set requires_grad flags
        for name, p in refined_model.named_parameters():
            p.requires_grad = True
            print(name, p.requires_grad)

        if hasattr(refined_model, "wrist_refine"):
            for n, p in refined_model.wrist_refine.named_parameters():
                p.requires_grad = True

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, refined_model.parameters()), lr=opts.lr)
        found2a, start_epoch2a, best_acc2a = load_checkpoint_for_phase(phase, refined_model, optimizer, opts.model_dir, device, load_optimizer_if_present=True)
        logger.info(f"Resuming Phase {phase} from epoch {start_epoch2a}, best_acc={best_acc2a:.4f}")
    else:
        # No Phase2a ckpt: initialize refined_model from Phase1 weights (copy backbone)
        # Load phase1 checkpoint if exists (non-strict to allow missing wrist_refine keys)
        if os.path.exists(phase1_ckpt):
            logger.info(f"Loading Phase 1 backbone weights into refined model (non-strict)...")
            ck = torch.load(phase1_ckpt, map_location=device)
            state_dict_phase1 = ck.get('state_dict', ck)  # handle both formats
            # If DataParallel was used, keys might have 'module.' prefix -- adapt by using strict=False
            try:
                refined_model.load_state_dict(state_dict_phase1, strict=False)
            except RuntimeError as e:
                # try removing 'module.' prefix if mismatch
                new_sd = {}
                for k, v in state_dict_phase1.items():
                    nk = k.replace('module.', '') if k.startswith('module.') else k
                    new_sd[nk] = v
                refined_model.load_state_dict(new_sd, strict=False)
        else:
            logger.info("No Phase 1 checkpoint exists to initialize Phase 2a; starting refined model from scratch.")

        # Freeze backbone, unfreeze refine branch
        for name, p in refined_model.named_parameters():
            p.requires_grad = False
        if hasattr(refined_model, "wrist_refine"):
            for n, p in refined_model.wrist_refine.named_parameters():
                p.requires_grad = True

        optimizer = Adam(filter(lambda p: p.requires_grad, refined_model.parameters()), lr=opts.lr)
        start_epoch2a = 0
        best_acc2a = 0.0
        logger.info("Starting Phase 2a (warm-up) with backbone frozen and wrist refinement trainable.")

    # Now train Phase 2a using refined_model
    model = refined_model  # use this for subsequent training/validation calls
    for epoch in range(start_epoch2a, start_epoch2a + 70):
        rst_trn = train(train_loader, SLP_rd_train, model, criterion, optimizer, epoch,
                        n_iter=opts.trainIter, logger=logger, opts=opts, visualizer=visualizer, device=device, phase=phase)
        rst_test = validate(test_loader, SLP_rd_test, model, criterion,
                            n_iter=opts.trainIter, logger=logger, opts=opts, device=device, phase=phase)
        pck_all = rst_test['pck']
        titles = list(SLP_rd_test.joints_name[:SLP_rd_test.joint_num_ori]) + ['total']
        pckh05 = np.array(pck_all)[:, -1]
        ut.prt_rst([pckh05], titles, ['pckh0.5'], fn_prt=logger.info)

        right_wrist_idx = SLP_rd_test.joints_name.index('R_Wrist')
        left_wrist_idx = SLP_rd_test.joints_name.index('L_Wrist')
        right_acc = pck_all[right_wrist_idx][-1]
        left_acc = pck_all[left_wrist_idx][-1]

        logger.info(f"Phase 2a Epoch {epoch}: Refined Wrist Acc R={rst_test['acc_right']:.3f}, L={rst_test['acc_left']:.3f}")
        

        # Use refined wrist accuracy from rst_test
        acc_right = rst_test['acc_right']
        acc_left = rst_test['acc_left']

        is_best = max(acc_right, acc_left) > best_acc2a
        best_acc2a = max(best_acc2a, acc_right, acc_left)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            'best_acc': best_acc2a,
            'acc_right': acc_right,
            'acc_left': acc_left,
            'optimizer': optimizer.state_dict(),
        }

        save_checkpoint(state, is_best, phase, opts.model_dir, logger)


    # ---------------------------
    # Phase 2b: Joint fine-tuning (unfreeze all)
    # ---------------------------
    phase = "2b"
    opts.phase = phase
    ck2b_path = os.path.join(opts.model_dir, f'checkpoint_phase{phase}.pth')
    # debug folder
    debug_dir = os.path.join(opts.model_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # At this point `model` is the refined_model (from Phase2a steps).
    # If there's an existing Phase2b checkpoint, resume from it (will overwrite model+optimizer).
    if os.path.exists(ck2b_path):
        # unfreeze all first to construct optimizer
        for name, p in model.named_parameters():
            p.requires_grad = True
        optimizer = Adam(model.parameters(), lr=opts.lr * 0.5)
        found2b, start_epoch2b, best_acc2b = load_checkpoint_for_phase(phase, model, optimizer, opts.model_dir, device, load_optimizer_if_present=True)
        logger.info(f"Resuming Phase {phase} from epoch {start_epoch2b}, best_acc={best_acc2b:.4f}")
        start_epoch = start_epoch2b
    else:
        # No ckpt: ensure backbone is unfrozen and optimizer created
        for name, p in model.named_parameters():
            p.requires_grad = True
        optimizer = Adam(model.parameters(), lr=opts.lr * 0.5)
        start_epoch = 0
        best_acc2b = 0.0
        logger.info("Starting Phase 2b (joint fine-tune) from latest available weights.")

    # Phase 2b training loop (same as before)
    for epoch in range(start_epoch, opts.end_epoch):
        rst_trn = train(train_loader, SLP_rd_train, model, criterion, optimizer, epoch,
                        n_iter=opts.trainIter, logger=logger, opts=opts, visualizer=visualizer, device=device, phase=phase)
        rst_test = validate(test_loader, SLP_rd_test, model, criterion,
                            n_iter=opts.trainIter, logger=logger, opts=opts, device=device, phase=phase)
        pck_all = rst_test['pck']
        titles = list(SLP_rd_test.joints_name[:SLP_rd_test.joint_num_ori]) + ['total']
        pckh05 = np.array(pck_all)[:, -1]
        ut.prt_rst([pckh05], titles, ['pckh0.5'], fn_prt=logger.info)
        right_wrist_idx = SLP_rd_test.joints_name.index('R_Wrist')
        left_wrist_idx = SLP_rd_test.joints_name.index('L_Wrist')
        right_acc = pck_all[right_wrist_idx][-1]
        left_acc = pck_all[left_wrist_idx][-1]

        is_best = max(right_acc, left_acc) > best_acc2b
        best_acc2b = max(best_acc2b, right_acc, left_acc)

        logger.info(f"Epoch {epoch}: Refined Wrist Acc R={rst_test['acc_right']:.3f}, L={rst_test['acc_left']:.3f}")

        state = {
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            'best_acc': best_acc2b,
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, phase, opts.model_dir, logger)

    # Final save and final test (unchanged)
    final_model_state_file = os.path.join(opts.model_dir, 'final_state.pth')
    final_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(final_state_dict, final_model_state_file)
    logger.info(f"=> saved final model to {final_model_state_file}")

    logger.info('----run final test----')
    rst_test = validate(
        test_loader, SLP_rd_test, model, criterion,
        n_iter=opts.trainIter, logger=logger, opts=opts, device=device, phase=phase, if_svVis=True
    )
    
    pck_all = rst_test['pck']
    pckh05 = np.array(pck_all)[:, -1]
    titles_c = list(SLP_rd_test.joints_name[:SLP_rd_test.joint_num_ori]) + ['total']
    ut.prt_rst([pckh05], titles_c, ['pckh0.5'], fn_prt=logger.info)

    pth_rst = path.join(opts.rst_dir, opts.nmTest + '.json')
    with open(pth_rst, 'w') as f:
        json.dump(rst_test, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# ---------- END REPLACEMENT PHASE BLOCKS ----------




if __name__ == '__main__':
    main()