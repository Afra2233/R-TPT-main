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
    k = 20

    if sim_matrix.dim() == 3:
        sim_matrix = sim_matrix.squeeze(0)

    sim_matrix = sim_matrix.clone()

    n = sim_matrix.size(0)
    k = min(k, max(n - 1, 1))

    eye = torch.eye(n, device=sim_matrix.device, dtype=torch.bool)
    sim_matrix[eye] = float('-inf')

    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    top_k_mean = top_k_values.mean(dim=-1)

    return top_k_mean  # [N]

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


def dirichlet_conservative_entropy(
    outputs,
    dir_temp=1.0,
    alpha_offset=1.0,
    gate_tau=0.1,
    lambda_cons=1.0
):
    """
    Dirichlet conservative entropy for R-TPT.

    outputs: [K, C], selected augmented-view logits.

    If selected views agree:
        minimize entropy as original R-TPT.
    If selected views disagree:
        reduce entropy minimization and enforce consistency to anchor.
    """
    alpha = logits_to_dirichlet_alpha(
        outputs,
        dir_temp=dir_temp,
        alpha_offset=alpha_offset
    )  # [K, C]

    # Dirichlet mean prediction
    alpha0 = alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
    probs = (alpha / alpha0).clamp_min(1e-8)  # [K, C]
    log_probs = torch.log(probs)

    # Pointwise entropy on Dirichlet mean
    ent_each = -(probs * log_probs).sum(dim=1)  # [K]
    ent_loss = ent_each.mean()

    # Anchor = average Dirichlet mean prediction, detached
    with torch.no_grad():
        anchor = probs.mean(dim=0, keepdim=True)
        anchor = anchor / anchor.sum(dim=1, keepdim=True).clamp_min(1e-8)
        anchor = anchor.clamp_min(1e-8)

    # View disagreement: KL(p_i || p_anchor)
    kl_each = F.kl_div(
        log_probs,
        anchor.expand_as(probs),
        reduction='none'
    ).sum(dim=1)  # [K]

    disagreement = kl_each.mean()

    # Gate:
    # high disagreement -> small gate -> less entropy minimization
    # low disagreement  -> large gate -> normal entropy minimization
    gate = torch.exp(-disagreement.detach() / gate_tau)
    gate = torch.clamp(gate, min=0.0, max=1.0)

    consistency_loss = kl_each.mean()

    loss = gate * ent_loss + lambda_cons * (1.0 - gate) * consistency_loss

    return loss, ent_loss.detach(), consistency_loss.detach(), gate.detach(), disagreement.detach()

def reliability_weighted_entropy(outputs, reliability):
    """
    COME-style conservative entropy for R-TPT.

    outputs: [K, C], selected view logits
    reliability: [K], reliability scores of selected views
    """
    probs = outputs.softmax(dim=1)
    log_probs = outputs.log_softmax(dim=1)
    ent = -(probs * log_probs).sum(dim=1)  # [K]

    w = reliability.view(-1).detach().float()

    # normalize reliability into positive weights
    w = w - w.min()
    w = w / (w.max() + 1e-6)

    # avoid zero-sum / all-zero weights
    w = w + 1e-3
    w = w / w.sum()

    loss = (w * ent).sum()
    return loss
def compute_ece(confidences, corrects, n_bins=15):
    """
    confidences: Tensor [N], max softmax probability
    corrects: Tensor [N], bool or 0/1
    return ECE in percentage
    """
    confidences = confidences.float()
    corrects = corrects.float()

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
            acc_in_bin = corrects[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return ece.item() * 100.0


def safe_mean(x):
    if x.numel() == 0:
        return float("nan")
    return x.float().mean().item()


def summarize_calibration_stats(
    clip_conf,
    clip_correct,
    tta_conf,
    tta_correct,
    high_conf_th=0.9,
    n_bins=15
):
    """
    Build sample-level analysis stats for CLIP vs R-TPT.
    All confidence tensors should be in [0,1].
    """
    clip_correct = clip_correct.bool()
    tta_correct = tta_correct.bool()

    clip_acc = clip_correct.float().mean().item() * 100.0
    tta_acc = tta_correct.float().mean().item() * 100.0

    clip_ece = compute_ece(clip_conf, clip_correct, n_bins=n_bins)
    tta_ece = compute_ece(tta_conf, tta_correct, n_bins=n_bins)

    conf_jump = tta_conf - clip_conf

    clip_wrong = ~clip_correct
    tta_wrong = ~tta_correct

    stats = {
        "clip_acc": clip_acc,
        "tta_acc": tta_acc,
        "acc_gain": tta_acc - clip_acc,

        "clip_ece": clip_ece,
        "tta_ece": tta_ece,
        "ece_change": tta_ece - clip_ece,

        "clip_avg_conf": clip_conf.float().mean().item() * 100.0,
        "tta_avg_conf": tta_conf.float().mean().item() * 100.0,
        "avg_conf_jump": conf_jump.float().mean().item() * 100.0,

        "clip_gap": clip_conf.float().mean().item() * 100.0 - clip_acc,
        "tta_gap": tta_conf.float().mean().item() * 100.0 - tta_acc,
        "gap_change": (
            tta_conf.float().mean().item() * 100.0 - tta_acc
        ) - (
            clip_conf.float().mean().item() * 100.0 - clip_acc
        ),

        "clip_wrong_conf": safe_mean(clip_conf[clip_wrong]) * 100.0,
        "tta_wrong_conf": safe_mean(tta_conf[tta_wrong]) * 100.0,

        "clip_correct_conf": safe_mean(clip_conf[clip_correct]) * 100.0,
        "tta_correct_conf": safe_mean(tta_conf[tta_correct]) * 100.0,

        "conf_jump_correct": safe_mean(conf_jump[tta_correct]) * 100.0,
        "conf_jump_wrong": safe_mean(conf_jump[tta_wrong]) * 100.0,

        "clip_wrong_high_conf_rate_all": (
            ((clip_conf >= high_conf_th) & clip_wrong).float().mean().item() * 100.0
        ),
        "tta_wrong_high_conf_rate_all": (
            ((tta_conf >= high_conf_th) & tta_wrong).float().mean().item() * 100.0
        ),

        "clip_wrong_high_conf_rate_wrong_only": (
            ((clip_conf[clip_wrong] >= high_conf_th).float().mean().item() * 100.0)
            if clip_wrong.sum().item() > 0 else float("nan")
        ),
        "tta_wrong_high_conf_rate_wrong_only": (
            ((tta_conf[tta_wrong] >= high_conf_th).float().mean().item() * 100.0)
            if tta_wrong.sum().item() > 0 else float("nan")
        ),
    }

    return stats


def format_stats_for_log(stats, mode_name):
    lines = []
    lines.append("========== Calibration Analysis ({}) ==========".format(mode_name))

    lines.append(
        "CLIP Acc: {:.4f} / R-TPT Acc: {:.4f} / Acc Gain: {:+.4f}".format(
            stats["clip_acc"], stats["tta_acc"], stats["acc_gain"]
        )
    )

    lines.append(
        "CLIP ECE: {:.4f} / R-TPT ECE: {:.4f} / ECE Change: {:+.4f}".format(
            stats["clip_ece"], stats["tta_ece"], stats["ece_change"]
        )
    )

    lines.append(
        "CLIP AvgConf: {:.4f} / R-TPT AvgConf: {:.4f} / Avg Conf Jump: {:+.4f}".format(
            stats["clip_avg_conf"], stats["tta_avg_conf"], stats["avg_conf_jump"]
        )
    )

    lines.append(
        "CLIP Conf-Acc Gap: {:.4f} / R-TPT Conf-Acc Gap: {:.4f} / Gap Change: {:+.4f}".format(
            stats["clip_gap"], stats["tta_gap"], stats["gap_change"]
        )
    )

    lines.append(
        "CLIP WrongConf: {:.4f} / R-TPT WrongConf: {:.4f}".format(
            stats["clip_wrong_conf"], stats["tta_wrong_conf"]
        )
    )

    lines.append(
        "CLIP CorrectConf: {:.4f} / R-TPT CorrectConf: {:.4f}".format(
            stats["clip_correct_conf"], stats["tta_correct_conf"]
        )
    )

    lines.append(
        "ConfJump on Correct R-TPT samples: {:+.4f} / ConfJump on Wrong R-TPT samples: {:+.4f}".format(
            stats["conf_jump_correct"], stats["conf_jump_wrong"]
        )
    )

    lines.append(
        "Wrong&HighConf sample rate all samples: CLIP {:.4f}% / R-TPT {:.4f}%".format(
            stats["clip_wrong_high_conf_rate_all"],
            stats["tta_wrong_high_conf_rate_all"]
        )
    )

    lines.append(
        "HighConf among wrong samples: CLIP {:.4f}% / R-TPT {:.4f}%".format(
            stats["clip_wrong_high_conf_rate_wrong_only"],
            stats["tta_wrong_high_conf_rate_wrong_only"]
        )
    )

    return "\n".join(lines)
def test_time_tuning(model, inputs, optimizer, scaler, args, reliability=None):
    """
    Original R-TPT:
        select confident views, then minimize mean entropy.

    Reliability-conservative R-TPT:
        keep the same confident-view selection,
        but weight selected views by R-TPT reliability score.
    """
    selected_idx = None

    for j in range(args.tta_steps):
        output_full = model(inputs)  # [N, C]

        if selected_idx is not None:
            output = output_full[selected_idx]

            if reliability is not None:
                rel_selected = reliability[selected_idx]
            else:
                rel_selected = None
        else:
            output, selected_idx = select_confident_samples(
                output_full,
                args.selection_p
            )
def test_time_tuning(model, inputs, optimizer, scaler, args, reliability=None):
    """
    Original R-TPT:
        select confident views, then minimize mean entropy.

    Dirichlet-conservative R-TPT:
        keep R-TPT confident-view selection,
        but replace aggressive entropy minimization with
        Dirichlet consensus-gated conservative entropy.
    """
    selected_idx = None

    for j in range(args.tta_steps):
        output_full = model(inputs)  # [N, C]

        if selected_idx is not None:
            output = output_full[selected_idx]
        else:
            output, selected_idx = select_confident_samples(
                output_full,
                args.selection_p
            )

        if args.dir_conservative_entropy:
            loss, loss_ent, loss_cons, gate, disagreement = dirichlet_conservative_entropy(
                output,
                dir_temp=args.dir_temp,
                alpha_offset=args.alpha_offset,
                gate_tau=args.dir_gate_tau,
                lambda_cons=args.lambda_cons
            )

            if args.debug_dir_conservative:
                print(
                    "[DirCons] "
                    "loss={:.6f}, ent={:.6f}, cons={:.6f}, gate={:.6f}, disagree={:.6f}".format(
                        loss.item(),
                        loss_ent.item(),
                        loss_cons.item(),
                        gate.item(),
                        disagreement.item()
                    )
                )

        elif args.rel_weighted_entropy and reliability is not None:
            rel_selected = reliability[selected_idx]
            loss = reliability_weighted_entropy(output, rel_selected)

        else:
            loss = entropy_avg(output)

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
        
        results = test_time_adapt_eval(
            val_loader,
            model,
            model_state,
            optimizer,
            optim_state,
            scaler,
            args,
            data_transform
        )

        del val_dataset, val_loader

        stats = results[2]

        if args.eps <= 0:
            print_log = (
                "=> Acc. on testset [{}]: "
                "Clip Clean Acc @1 {:.4f} / Rtpt Clean Acc @1 {:.4f} / "
                "Clip Clean ECE {:.4f} / Rtpt Clean ECE {:.4f} / "
                "Avg Conf Jump {:+.4f} / WrongConf {:.4f}"
            ).format(
                dset,
                stats["clip_acc"],
                stats["tta_acc"],
                stats["clip_ece"],
                stats["tta_ece"],
                stats["avg_conf_jump"],
                stats["tta_wrong_conf"]
            )

            save_log = {
                "clean_acc": stats["clip_acc"],
                "tta_clean_acc": stats["tta_acc"],
                "clean_ece": stats["clip_ece"],
                "tta_clean_ece": stats["tta_ece"],
                "avg_conf_jump": stats["avg_conf_jump"],
                "tta_wrong_conf": stats["tta_wrong_conf"],
                "tta_wrong_high_conf_rate_all": stats["tta_wrong_high_conf_rate_all"],
                "tta_wrong_high_conf_rate_wrong_only": stats["tta_wrong_high_conf_rate_wrong_only"],
                "all_stats": stats,
            }

        else:
            print_log = (
                "=> Acc. on testset [{}]: "
                "Clip Robust Acc @1 {:.4f} / Rtpt Robust Acc @1 {:.4f} / "
                "Clip Robust ECE {:.4f} / Rtpt Robust ECE {:.4f} / "
                "Acc Gain {:+.4f} / ECE Change {:+.4f} / "
                "Rtpt WrongConf {:.4f} / Rtpt WrongHighConf {:.4f}%"
            ).format(
                dset,
                stats["clip_acc"],
                stats["tta_acc"],
                stats["clip_ece"],
                stats["tta_ece"],
                stats["acc_gain"],
                stats["ece_change"],
                stats["tta_wrong_conf"],
                stats["tta_wrong_high_conf_rate_wrong_only"]
            )

            save_log = {
                "robust_acc": stats["clip_acc"],
                "tta_robust_acc": stats["tta_acc"],
                "robust_ece": stats["clip_ece"],
                "tta_robust_ece": stats["tta_ece"],
                "acc_gain": stats["acc_gain"],
                "ece_change": stats["ece_change"],
                "tta_wrong_conf": stats["tta_wrong_conf"],
                "tta_wrong_high_conf_rate_all": stats["tta_wrong_high_conf_rate_all"],
                "tta_wrong_high_conf_rate_wrong_only": stats["tta_wrong_high_conf_rate_wrong_only"],
                "all_stats": stats,
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
    clip_pred_list = []

    tta_conf_list = []
    tta_correct_list = []
    tta_pred_list = []

    label_list = []

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, tpt1],
        prefix='Test: '
    )

    model.eval()

    if args.eps > 0.0:
        assert args.steps > 0
        atk = torchattacks.PGD(
            model,
            eps=args.eps / 255,
            alpha=args.alpha / 255,
            steps=args.steps
        )

    end = time.time()

    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)

        if args.eps > 0.0:
            image = images[0].cuda(args.gpu, non_blocking=True)
            adv_image = atk(image, target)

            img_adv = transforms.ToPILImage()(adv_image.squeeze(0).detach().cpu())
            images = data_transform(img_adv)
            images = [_.unsqueeze(0) for _ in images]

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images

        images = torch.cat(images, dim=0)

        with torch.no_grad():
            model.reset()

        optimizer.load_state_dict(optim_state)

        with torch.no_grad():
            clip_output = model(image)
            clip_features, _, _ = model.forward_features(images)

        sim_matrix_images = torch.bmm(
            clip_features.unsqueeze(0),
            clip_features.unsqueeze(0).permute(0, 2, 1)
        )

        score = get_top_sim(sim_matrix_images).view(-1)  # [N]
# =============================================================
        assert args.tta_steps > 0
        test_time_tuning(
            model,
            images,
            optimizer,
            scaler,
            args,
            reliability=score
        )
        # if args.tta_steps > 0:
        #     test_time_tuning(
        #         model,
        #         images,
        #         optimizer,
        #         scaler,
        #         args,
        #         reliability=score
        #     )
# =============================================================
        with torch.no_grad():
            tuned_outputs = model(images)

        weight = torch.nn.functional.softmax(score / 0.01, dim=-1)

        tta_output = torch.sum(
            weight.unsqueeze(-1) * tuned_outputs,
            dim=0,
            keepdim=True
        )
        # tta_output = torch.bmm(
        #     weight.unsqueeze(-1).transpose(1, 2),
        #     tuned_outputs.unsqueeze(0)
        # ).squeeze(1)

        acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))
        tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        tpt1.update(tpt_acc1[0], images.size(0))

        # ============================================================
        # Per-sample calibration records
        # ============================================================
        with torch.no_grad():
            clip_prob = torch.softmax(clip_output, dim=1)
            clip_conf, clip_pred = clip_prob.max(dim=1)
            clip_correct = clip_pred.eq(target)

            tta_prob = torch.softmax(tta_output, dim=1)
            tta_conf, tta_pred = tta_prob.max(dim=1)
            tta_correct = tta_pred.eq(target)

            clip_conf_list.append(clip_conf.detach().cpu())
            clip_correct_list.append(clip_correct.detach().cpu())
            clip_pred_list.append(clip_pred.detach().cpu())

            tta_conf_list.append(tta_conf.detach().cpu())
            tta_correct_list.append(tta_correct.detach().cpu())
            tta_pred_list.append(tta_pred.detach().cpu())

            label_list.append(target.detach().cpu())

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(val_loader):
            if args.eps <= 0:
                print_log = 'iter:{}/{}, clip_acc1={}, tta_acc1={}'.format(
                    i, len(val_loader), top1.avg, tpt1.avg
                )
            else:
                print_log = 'iter:{}/{}, clip_adv1={}, tta_adv1={}'.format(
                    i, len(val_loader), top1.avg, tpt1.avg
                )

            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log + '\n')
            progress.display(i)

    progress.display_summary()

    # ============================================================
    # Aggregate calibration analysis
    # ============================================================
    clip_conf_all = torch.cat(clip_conf_list)
    clip_correct_all = torch.cat(clip_correct_list)
    clip_pred_all = torch.cat(clip_pred_list)

    tta_conf_all = torch.cat(tta_conf_list)
    tta_correct_all = torch.cat(tta_correct_list)
    tta_pred_all = torch.cat(tta_pred_list)

    label_all = torch.cat(label_list)

    mode_name = "Clean" if args.eps <= 0 else "Robust"

    stats = summarize_calibration_stats(
        clip_conf=clip_conf_all,
        clip_correct=clip_correct_all,
        tta_conf=tta_conf_all,
        tta_correct=tta_correct_all,
        high_conf_th=args.high_conf_th,
        n_bins=args.ece_bins
    )

    analysis_log = format_stats_for_log(stats, mode_name)

    args.out_file.write(analysis_log + '\n')
    args.out_file.flush()
    print(analysis_log + '\n')

    # ============================================================
    # Save detailed per-sample stats for later plotting
    # ============================================================
    save_log = {
        "mode": mode_name,
        "clip_conf": clip_conf_all.numpy(),
        "clip_correct": clip_correct_all.numpy(),
        "clip_pred": clip_pred_all.numpy(),
        "tta_conf": tta_conf_all.numpy(),
        "tta_correct": tta_correct_all.numpy(),
        "tta_pred": tta_pred_all.numpy(),
        "label": label_all.numpy(),
        "stats": stats,
    }

    torch.save(save_log, os.path.join(args.output_dir, "calibration_analysis.pt"))

    return [stats["clip_acc"], stats["tta_acc"], stats]


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

    parser.add_argument('--ece_bins', type=int, default=15,
                        help='number of bins for ECE computation')

    parser.add_argument('--high_conf_th', type=float, default=0.9,
                        help='threshold for high-confidence wrong predictions')
    parser.add_argument('--rel_weighted_entropy', action='store_true', default=False,
                    help='use R-TPT reliability-weighted conservative entropy loss')
    
    parser.add_argument('--dir_conservative_entropy', action='store_true', default=False,
                    help='use Dirichlet consensus-gated conservative entropy loss')

    parser.add_argument('--dir_temp', type=float, default=1.0,
                        help='temperature for mapping logits to Dirichlet alpha')

    parser.add_argument('--alpha_offset', type=float, default=1.0,
                        help='offset added to Dirichlet alpha')

    parser.add_argument('--dir_gate_tau', type=float, default=0.1,
                        help='temperature for Dirichlet disagreement gate')

    parser.add_argument('--lambda_cons', type=float, default=1.0,
                        help='weight for conservative consistency loss')

    parser.add_argument('--debug_dir_conservative', action='store_true', default=False,
                        help='print Dirichlet conservative entropy debug info')
    main()
