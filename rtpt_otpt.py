import argparse
import time
from copy import deepcopy
from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models
import torch.nn.functional as F

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import os

import torchattacks

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_top_sim(sim_matrix):
    k = 20 # use 20 neighbor
    sim_matrix[sim_matrix>=1.0] = float('-inf')
    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    top_k_mean = top_k_values.mean(dim=-1)
    return top_k_mean

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def entropy_avg(outputs):
    batch_entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
    return batch_entropy.mean()

def logits_to_dirichlet_alpha(logits, dir_temp=1.0, alpha_offset=1.0):
    """
    Map logits to Dirichlet concentration parameters.
    alpha = softplus(logits / dir_temp) + alpha_offset
    """
    logits = logits.float()
    scaled_logits = torch.clamp(logits / dir_temp, min=-20.0, max=20.0)
    alpha = F.softplus(scaled_logits) + alpha_offset
    alpha = alpha.clamp_min(1e-6)
    return alpha


def dirichlet_kl(alpha_p, alpha_q, reduction='batchmean'):
    """
    KL( Dir(alpha_p) || Dir(alpha_q) )
    alpha_p, alpha_q: [N, C]
    """
    alpha_p = alpha_p.float().clamp_min(1e-6)
    alpha_q = alpha_q.float().clamp_min(1e-6)

    sum_p = alpha_p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    sum_q = alpha_q.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    term1 = torch.lgamma(sum_p) - torch.lgamma(alpha_p).sum(dim=-1, keepdim=True)
    term2 = torch.lgamma(sum_q) - torch.lgamma(alpha_q).sum(dim=-1, keepdim=True)
    term3 = ((alpha_p - alpha_q) * (
        torch.digamma(alpha_p) - torch.digamma(sum_p)
    )).sum(dim=-1, keepdim=True)

    kl = term1 - term2 + term3
    kl = kl.squeeze(-1)
    kl = torch.nan_to_num(kl, nan=0.0, posinf=1e4, neginf=0.0)

    if reduction in ['mean', 'batchmean']:
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    elif reduction == 'none':
        return kl
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def dirichlet_center_consistency_loss(logits, dir_temp=1.0, alpha_offset=1.0):
    """
    Dirichlet consistency across selected augmented views.
    logits: [N, C]
    """
    alpha = logits_to_dirichlet_alpha(
        logits,
        dir_temp=dir_temp,
        alpha_offset=alpha_offset
    )

    center_alpha = alpha.mean(dim=0, keepdim=True)
    center_alpha = center_alpha.expand_as(alpha).contiguous()

    loss_dir = dirichlet_kl(alpha, center_alpha, reduction='batchmean')
    loss_dir = torch.nan_to_num(loss_dir, nan=0.0, posinf=1e4, neginf=0.0)

    return loss_dir, alpha


def evidence_penalty(alpha, mode='mean_total'):
    """
    Penalize excessive total evidence.
    """
    alpha = alpha.float().clamp_min(1e-6)
    alpha0 = alpha.sum(dim=-1)

    if mode == 'mean_total':
        out = alpha0.mean()
    elif mode == 'log_total':
        out = torch.log(alpha0 + 1e-6).mean()
    else:
        raise ValueError(f"Unknown evidence penalty mode: {mode}")

    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=0.0)

def compute_ece(confidences, corrects, n_bins=15):
    """
    confidences: Tensor, shape [N], max softmax probability
    corrects: Tensor, shape [N], 0/1 correctness
    return: ECE in percentage, e.g. 3.25 means 3.25%
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=confidences.device)
    ece = torch.zeros(1, device=confidences.device)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = corrects[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item() * 100.0

def otpt_orthogonality_loss(model, gpu):
    """
    O-TPT orthogonality regularization on text features.
    Requires model.textfeatures_ to be available after model(inputs).
    """
    if not hasattr(model, "textfeatures_"):
        raise AttributeError(
            "model.textfeatures_ not found. "
            "Please check whether your custom_clip model stores text features as model.textfeatures_."
        )

    text_feature = model.textfeatures_  # [C, D]

    Wwt = torch.matmul(text_feature, text_feature.T)  # [C, C]

    e = torch.eye(Wwt.shape[1], device=Wwt.device)

    M_norm = torch.linalg.norm(Wwt, dim=0, keepdim=True)
    scaled_e = e * M_norm

    u = Wwt - scaled_e
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True) + 1e-8
    v = u / u_norm

    normalized_matrix_exp = v.unsqueeze(2)
    normalized_matrix_T_exp = v.unsqueeze(1)

    outer_products = normalized_matrix_exp @ normalized_matrix_T_exp
    scaled_matrix = 2 * outer_products

    identity_matrix_dim = e.unsqueeze(0).expand(Wwt.shape[1], -1, -1)

    transformed_matrix = identity_matrix_dim - scaled_matrix

    Wwt_exp = Wwt.unsqueeze(2)
    Hx = torch.bmm(transformed_matrix, Wwt_exp)
    Hx = Hx.squeeze(2)

    Ht_ortho = Hx - e
    Ht_ortho_norm = torch.linalg.norm(Ht_ortho, dim=-1)

    return Ht_ortho_norm.mean()
# def test_time_tuning(model, inputs, optimizer, scaler, args):
    
#     selected_idx = None
#     for j in range(args.tta_steps):
#         if True:
#             output = model(inputs) 

#             if selected_idx is not None:
#                 output = output[selected_idx]
#             else:
#                 output, selected_idx = select_confident_samples(output, args.selection_p)

#             loss = entropy_avg(output)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     return
def test_time_tuning(model, inputs, optimizer, scaler, args):
    selected_idx = None

    for j in range(args.tta_steps):
        output_full = model(inputs)

        if selected_idx is not None:
            output = output_full[selected_idx]
        else:
            output, selected_idx = select_confident_samples(
                output_full,
                args.selection_p
            )

        # Original R-TPT entropy loss
        loss_ent = entropy_avg(output)
        loss = args.lambda_tpt * loss_ent

        loss_dir = None
        loss_evi = None
        loss_otpt = None

        # Dirichlet view consistency
        if args.dirichlet_consistency:
            loss_dir, alpha = dirichlet_center_consistency_loss(
                output.float(),
                dir_temp=args.dir_temp,
                alpha_offset=args.alpha_offset
            )

            loss = loss + args.lambda_dir * loss_dir

            if args.evidence_penalty:
                loss_evi = evidence_penalty(
                    alpha,
                    mode=args.evidence_mode
                )
                loss = loss + args.lambda_evi * loss_evi

        # O-TPT orthogonality loss
        if args.otpt:
            loss_otpt = otpt_orthogonality_loss(model, args.gpu)
            loss = loss + args.lambda_term * loss_otpt

        if args.debug_dirichlet:
            print(
                f"[TTA step {j}] "
                f"loss={loss.item():.6f}, "
                f"ent={loss_ent.item():.6f}"
            )

            if loss_dir is not None:
                print(
                    f"[TTA step {j}] "
                    f"dir={loss_dir.item():.6f}, "
                    f"alpha_min={alpha.min().item():.6f}, "
                    f"alpha_max={alpha.max().item():.6f}"
                )

            if loss_evi is not None:
                print(f"[TTA step {j}] evi={loss_evi.item():.6f}")

            if loss_otpt is not None:
                print(f"[TTA step {j}] otpt={loss_otpt.item():.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    args.alpha = args.eps / 4.0
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets, 'eps_'+str(args.eps)+'_alpha_'+str(args.alpha)+'_step_'+str(args.steps))

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(os.path.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    assert args.gpu is not None

    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # model
    dset = args.test_sets
    if len(dset) > 1: 
        classnames = eval("{}_classes".format(dset.lower()))
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if dset in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(dset.lower()))
            if dset == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    args.classnames = classnames

    model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
    model_state = None

    ###### load robust vision encoder (TeCoA) ######
    if len(args.load_tecoa) > 0:
        args.robust_pretrain_path = {
            'RN50-eps1': 'pretrain/tecoa/rn50_eps1.pth.tar',
        }[args.load_tecoa]
        robust_state_dict = torch.load(args.robust_pretrain_path, map_location='cpu')
        model.image_encoder.load_state_dict(robust_state_dict['vision_encoder_state_dict'])
        print('load robust vision encoder')

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
                param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    scaler = None
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    
    results = {}
    if True:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # normalize
            ])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                        augmix=len(dset)>1)
        batchsize = 1

        val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

        print("evaluating: {}".format(dset))
        
        results = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform)
        del val_dataset, val_loader
        if args.eps <= 0:
            print_log = (
                "=> Acc. on testset [{}]: "
                "Clip Clean Acc @1 {:.4f} / Rtpt Clean Acc @1 {:.4f} / "
                "Clip Clean ECE {:.4f} / Rtpt Clean ECE {:.4f}"
            ).format(dset, results[0], results[1], results[2], results[3])

            save_log = {
                'clean_acc': results[0],
                'tta_clean_acc': results[1],
                'clean_ece': results[2],
                'tta_clean_ece': results[3],
            }
        else:
            print_log = (
                "=> Acc. on testset [{}]: "
                "Clip Robust Acc @1 {:.4f} / Rtpt Robust Acc @1 {:.4f} / "
                "Clip Robust ECE {:.4f} / Rtpt Robust ECE {:.4f}"
            ).format(dset, results[0], results[1], results[2], results[3])

            save_log = {
                'robust_acc': results[0],
                'tta_robust_acc': results[1],
                'robust_ece': results[2],
                'tta_robust_ece': results[3],
            }
      
        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log+'\n')

        torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    clip_conf_list = []
    clip_correct_list = []
    tta_conf_list = []
    tta_correct_list = []

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, tpt1],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()

    if args.eps > 0.0:
        assert args.steps > 0
        atk = torchattacks.PGD(model, eps=args.eps/255, alpha=args.alpha/255, steps=args.steps)
        
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)

        if args.eps > 0.0:
            image = images[0].cuda(args.gpu, non_blocking=True)
            adv_image = atk(image, target)        
            img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        
        images = torch.cat(images, dim=0)

        # reset model
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        with torch.no_grad():
            clip_output = model(image)
            clip_features, _, _ = model.forward_features(images)
            clip_outputs = model(images)

        assert args.tta_steps > 0
        test_time_tuning(model, images, optimizer, scaler, args)
        with torch.no_grad():
            tuned_outputs = model(images)
        
        sim_matrix_images = torch.bmm(clip_features.unsqueeze(0), clip_features.unsqueeze(0).permute(0, 2, 1))
        score = get_top_sim(sim_matrix_images)
        weight = torch.nn.functional.softmax(score/0.01, dim=-1)
        tta_output = torch.bmm(weight.unsqueeze(-1).transpose(1, 2), tuned_outputs.unsqueeze(0)).squeeze(1)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))
        tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))

        with torch.no_grad():
            clip_prob = torch.softmax(clip_output, dim=1)
            clip_conf, clip_pred = clip_prob.max(dim=1)
            clip_correct = clip_pred.eq(target)

            tta_prob = torch.softmax(tta_output, dim=1)
            tta_conf, tta_pred = tta_prob.max(dim=1)
            tta_correct = tta_pred.eq(target)

            clip_conf_list.append(clip_conf.detach().cpu())
            clip_correct_list.append(clip_correct.detach().cpu())

            tta_conf_list.append(tta_conf.detach().cpu())
            tta_correct_list.append(tta_correct.detach().cpu())
       
        top1.update(acc1[0], images.size(0))
        tpt1.update(tpt_acc1[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            if args.eps <= 0:
                print_log = 'iter:{}/{}, clip_acc1={}, tta_acc1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            else:
                print_log = 'iter:{}/{}, clip_adv1={}, tta_adv1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')
            progress.display(i)

    progress.display_summary()

    clip_conf_all = torch.cat(clip_conf_list)
    clip_correct_all = torch.cat(clip_correct_list)

    tta_conf_all = torch.cat(tta_conf_list)
    tta_correct_all = torch.cat(tta_correct_list)

    clip_ece = compute_ece(clip_conf_all, clip_correct_all, n_bins=15)
    tta_ece = compute_ece(tta_conf_all, tta_correct_all, n_bins=15)

    if args.eps <= 0:
        print_log = "Clip Clean ECE: {:.4f} / Rtpt Clean ECE: {:.4f}".format(clip_ece, tta_ece)
    else:
        print_log = "Clip Robust ECE: {:.4f} / Rtpt Robust ECE: {:.4f}".format(clip_ece, tta_ece)

    args.out_file.write(print_log + '\n')
    args.out_file.flush()
    print(print_log + '\n')

    return [top1.avg, tpt1.avg, clip_ece, tta_ece]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='Caltech101')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output_results/ckps/rtpt')

    parser.add_argument('--eps', default=0.0, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--steps', type=int, default=0)

    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')

    parser.add_argument('--load_tecoa', type=str, default='', choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'])
    parser.add_argument('--dirichlet_consistency', action='store_true', default=False,
                    help='use Dirichlet consistency across selected augmented views')

    parser.add_argument('--dir_temp', type=float, default=1.0,
                        help='temperature for mapping logits to Dirichlet alpha')

    parser.add_argument('--alpha_offset', type=float, default=1.0,
                        help='offset added to Dirichlet alpha')

    parser.add_argument('--lambda_dir', type=float, default=0.1,
                        help='weight for Dirichlet consistency loss')

    parser.add_argument('--lambda_tpt', type=float, default=1.0,
                        help='weight for original R-TPT entropy loss')

    parser.add_argument('--evidence_penalty', action='store_true', default=False,
                        help='penalize excessive total evidence')

    parser.add_argument('--lambda_evi', type=float, default=1e-4,
                        help='weight for evidence penalty')

    parser.add_argument('--evidence_mode', type=str, default='log_total',
                        choices=['mean_total', 'log_total'],
                        help='type of evidence penalty')

    parser.add_argument('--debug_dirichlet', action='store_true', default=False,
                        help='print debug information for Dirichlet branch')
    
    parser.add_argument('--otpt', action='store_true', default=False,
                    help='use O-TPT orthogonality regularization')

    parser.add_argument('--lambda_term', type=float, default=0.0,
                        help='weight for O-TPT orthogonality loss')
    main()
