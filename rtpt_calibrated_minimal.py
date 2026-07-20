import argparse
import math
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

def get_top_sim(sim_matrix, k=20):
    """Mean similarity to the K nearest *other* augmented views."""
    n_views = sim_matrix.size(-1)
    if n_views <= 1:
        return torch.ones(sim_matrix.shape[:-1], device=sim_matrix.device,
                          dtype=sim_matrix.dtype)

    k = min(k, n_views - 1)
    sim_matrix = sim_matrix.clone()
    diagonal = torch.eye(n_views, device=sim_matrix.device,
                         dtype=torch.bool).unsqueeze(0)
    sim_matrix = sim_matrix.masked_fill(diagonal, float('-inf'))
    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    return top_k_values.mean(dim=-1)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def entropy_per_sample(logits):
    """Entropy in nats for every augmented view."""
    return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)


def select_confident_indices(logits, top):
    """Keep the original R-TPT low-entropy view selection."""
    batch_entropy = entropy_per_sample(logits)
    n_select = max(1, int(batch_entropy.size(0) * top))
    return torch.argsort(batch_entropy, descending=False)[:n_select]


def js_divergence_to_mean(probs, eps=1e-8):
    """JS divergence between each view prediction and the mean prediction."""
    probs = probs.clamp_min(eps)
    mean_prob = probs.mean(dim=0, keepdim=True).clamp_min(eps)
    midpoint = 0.5 * (probs + mean_prob)

    kl_view = (probs * (probs.log() - midpoint.log())).sum(dim=-1)
    kl_mean = (mean_prob * (mean_prob.log() - midpoint.log())).sum(dim=-1)
    return 0.5 * (kl_view + kl_mean)


@torch.no_grad()
def compute_disagreement_control(base_logits, args):
    """Compute label-free view and sample controls before prompt tuning.

    Per-view control:
        views closer to the mean prediction receive larger weights.

    Per-sample control:
        an image with stronger cross-view disagreement receives a higher
        entropy floor and a smaller prompt-tuning learning rate.

    The model logits and CLIP/R-TPT temperature are never rescaled.
    """
    probs = base_logits.softmax(dim=-1)
    js = js_divergence_to_mean(probs)
    normalized_js = (js / math.log(2.0)).clamp(0.0, 1.0)

    # Relative weighting across views. Blend with uniform weighting so the
    # method can be ablated continuously with agreement_strength=0.
    agreement_weight = torch.softmax(
        -normalized_js / max(args.agreement_temp, 1e-8), dim=0)
    uniform_weight = torch.full_like(
        agreement_weight, 1.0 / agreement_weight.numel())
    view_weight = (
        args.agreement_strength * agreement_weight
        + (1.0 - args.agreement_strength) * uniform_weight
    )
    view_weight = view_weight / view_weight.sum().clamp_min(1e-8)

    # One uncertainty value for the current test image (all augmentations).
    sample_disagreement = normalized_js.mean()

    # High disagreement -> stop entropy minimization earlier.
    normalized_floor = (
        args.entropy_floor_base
        + args.entropy_floor_scale * sample_disagreement
    ).clamp(0.0, 1.0)

    # High disagreement -> a smaller actual AdamW step. Scaling the full loss
    # alone is not enough because Adam largely normalizes global gradient scale.
    lr_gate = (1.0 - sample_disagreement).clamp(0.0, 1.0)
    lr_gate = lr_gate.pow(args.sample_lr_power)
    lr_gate = lr_gate.clamp(min=args.sample_lr_min, max=1.0)

    return (
        view_weight.detach(),
        normalized_floor.detach(),
        lr_gate.detach(),
        sample_disagreement.detach(),
        normalized_js.detach(),
    )


def test_time_tuning(model, inputs, optimizer, scaler, args,
                     selected_idx, view_weight, normalized_floor, lr_gate):
    """Disagreement-controlled entropy minimization.

    This changes only prompt adaptation. It does not divide logits by a new
    temperature and does not modify R-TPT's original density ensemble.
    """
    selected_weight = view_weight[selected_idx]
    selected_weight = selected_weight / selected_weight.sum().clamp_min(1e-8)

    # Save/restore the configured LR because every test sample resets the
    # optimizer state but shares the same optimizer object.
    original_lrs = [group['lr'] for group in optimizer.param_groups]
    gate = float(lr_gate.item())
    for group, base_lr in zip(optimizer.param_groups, original_lrs):
        group['lr'] = base_lr * gate

    try:
        for _ in range(args.tta_steps):
            output = model(inputs)[selected_idx]
            entropy = entropy_per_sample(output)

            floor_nats = normalized_floor * math.log(output.size(-1))
            controlled_entropy = torch.relu(entropy - floor_nats)
            loss = (selected_weight * controlled_entropy).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    finally:
        for group, base_lr in zip(optimizer.param_groups, original_lrs):
            group['lr'] = base_lr


class ECEMeter:
    """Streaming Expected Calibration Error; returned value is a percentage."""
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.count = torch.zeros(n_bins, dtype=torch.float64)
        self.conf_sum = torch.zeros(n_bins, dtype=torch.float64)
        self.correct_sum = torch.zeros(n_bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, target):
        probs = logits.softmax(dim=-1)
        confidence, prediction = probs.max(dim=-1)
        correct = prediction.eq(target).to(torch.float64)

        confidence = confidence.detach().cpu().to(torch.float64)
        correct = correct.detach().cpu()
        bin_id = torch.clamp((confidence * self.n_bins).long(),
                             max=self.n_bins - 1)

        self.count += torch.bincount(bin_id, minlength=self.n_bins).to(torch.float64)
        self.conf_sum += torch.bincount(
            bin_id, weights=confidence, minlength=self.n_bins).to(torch.float64)
        self.correct_sum += torch.bincount(
            bin_id, weights=correct, minlength=self.n_bins).to(torch.float64)

    def compute(self):
        total = self.count.sum().clamp_min(1.0)
        nonempty = self.count > 0
        bin_conf = torch.zeros_like(self.conf_sum)
        bin_acc = torch.zeros_like(self.correct_sum)
        bin_conf[nonempty] = self.conf_sum[nonempty] / self.count[nonempty]
        bin_acc[nonempty] = self.correct_sum[nonempty] / self.count[nonempty]
        ece = ((self.count / total) * (bin_acc - bin_conf).abs()).sum()
        return 100.0 * ece.item()


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

    # Build the dataset once. When --eval_both is enabled, the same loader is
    # traversed once for clean inputs and once for adversarial inputs.
    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_transform = AugMixAugmenter(
        base_transform, preprocess, n_views=args.batch_size - 1,
        augmix=len(dset) > 1)

    val_dataset = build_dataset(
        dset, data_transform, args.data, mode=args.dataset_mode)
    print("number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.eval_both:
        if args.eps <= 0:
            raise ValueError('--eval_both requires --eps > 0 for the robust evaluation')
        eval_modes = [
            ('clean', 0.0, 0),
            ('adv', args.eps, args.steps),
        ]
    else:
        mode_name = 'clean' if args.eps <= 0 else 'adv'
        eval_modes = [(mode_name, args.eps, args.steps)]

    all_results = {}
    save_log = {}
    for mode_name, eval_eps, eval_steps in eval_modes:
        print("evaluating: {} [{}]".format(dset, mode_name))
        result = test_time_adapt_eval(
            val_loader, model, model_state, optimizer, optim_state, scaler,
            args, data_transform, eval_eps=eval_eps, eval_steps=eval_steps,
            mode_name=mode_name)
        all_results[mode_name] = result

        if mode_name == 'clean':
            print_log = (
                "=> Testset [{}]: Clean Acc @1 {:.3f} / TTA Clean Acc @1 {:.3f} / "
                "Clean ECE {:.3f} / TTA Clean ECE {:.3f}").format(
                    dset, result[0], result[1], result[2], result[3])
            save_log.update({
                'clean_acc': result[0],
                'tta_clean_acc': result[1],
                'clean_ece': result[2],
                'tta_clean_ece': result[3],
            })
        else:
            print_log = (
                "=> Testset [{}]: Robust Acc @1 {:.3f} / TTA Robust Acc @1 {:.3f} / "
                "Robust ECE {:.3f} / TTA Robust ECE {:.3f}").format(
                    dset, result[0], result[1], result[2], result[3])
            save_log.update({
                'robust_acc': result[0],
                'tta_robust_acc': result[1],
                'robust_ece': result[2],
                'tta_robust_ece': result[3],
                # Backward-compatible names.
                'adv_acc': result[0],
                'tta_adv_acc': result[1],
                'adv_ece': result[2],
                'tta_adv_ece': result[3],
            })

        args.out_file.write(print_log + '\n')
        args.out_file.flush()
        print(print_log + '\n')

    if args.eval_both:
        clean = all_results['clean']
        robust = all_results['adv']
        summary = (
            "\n========== FINAL CLEAN + ROBUST SUMMARY ==========\n"
            "Clean Accuracy  : zero-shot {:.3f} | TTA {:.3f}\n"
            "Robust Accuracy : zero-shot {:.3f} | TTA {:.3f}\n"
            "Clean ECE       : zero-shot {:.3f} | TTA {:.3f}\n"
            "Robust ECE      : zero-shot {:.3f} | TTA {:.3f}\n"
            "================================================\n"
        ).format(
            clean[0], clean[1], robust[0], robust[1],
            clean[2], clean[3], robust[2], robust[3])
        print(summary)
        args.out_file.write(summary + '\n')
        args.out_file.flush()

    torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))
    del val_dataset, val_loader


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform,
                         eval_eps=None, eval_steps=None, mode_name='eval'):
    if eval_eps is None:
        eval_eps = args.eps
    if eval_steps is None:
        eval_steps = args.steps

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    tpt1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    clip_ece_meter = ECEMeter(args.ece_bins)
    tta_ece_meter = ECEMeter(args.ece_bins)
    disagreement_meter = AverageMeter('Disagree', ':6.4f', Summary.AVERAGE)
    floor_meter = AverageMeter('EntFloor', ':6.4f', Summary.AVERAGE)
    lr_gate_meter = AverageMeter('LRGate', ':6.4f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, tpt1],
        prefix='{}: '.format(mode_name.capitalize()))

    # reset model and switch to evaluate mode
    model.eval()

    if eval_eps > 0.0:
        assert eval_steps > 0
        eval_alpha = eval_eps / 4.0
        atk = torchattacks.PGD(
            model, eps=eval_eps / 255.0, alpha=eval_alpha / 255.0,
            steps=eval_steps)
        
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)

        # PGD must start from the same reset prompt for every test sample.
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        if eval_eps > 0.0:
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

        # Compute disagreement controls once before prompt tuning. All controls
        # are detached and use no test labels.
        (view_weight, normalized_floor, lr_gate,
         sample_disagreement, view_disagreement) = compute_disagreement_control(
             clip_outputs, args)
        selected_idx = select_confident_indices(clip_outputs, args.selection_p)

        assert args.tta_steps > 0
        test_time_tuning(
            model, images, optimizer, scaler, args,
            selected_idx, view_weight, normalized_floor, lr_gate)
        with torch.no_grad():
            tuned_outputs = model(images)

        # Keep the original R-TPT density-based ensemble exactly. No extra
        # temperature scaling or logit division is applied here.
        sim_matrix_images = torch.bmm(
            clip_features.unsqueeze(0),
            clip_features.unsqueeze(0).permute(0, 2, 1))
        score = get_top_sim(sim_matrix_images, k=args.num_neighbors)
        weight = torch.nn.functional.softmax(score / args.density_temp, dim=-1)
        tta_output = torch.bmm(
            weight.unsqueeze(-1).transpose(1, 2),
            tuned_outputs.unsqueeze(0)).squeeze(1)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))
        tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))
        clip_ece_meter.update(clip_output, target)
        tta_ece_meter.update(tta_output, target)
        disagreement_meter.update(float(sample_disagreement.item()), target.size(0))
        floor_meter.update(float(normalized_floor.item()), target.size(0))
        lr_gate_meter.update(float(lr_gate.item()), target.size(0))

        top1.update(acc1[0], target.size(0))
        tpt1.update(tpt_acc1[0], target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            if eval_eps <= 0:
                print_log = 'iter:{}/{}, clip_clean1={}, tta_clean1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            else:
                print_log = 'iter:{}/{}, clip_robust1={}, tta_robust1={}'.format(i, len(val_loader), top1.avg, tpt1.avg)
            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log+'\n')
            progress.display(i)

    progress.display_summary()
    clip_ece = clip_ece_meter.compute()
    tta_ece = tta_ece_meter.compute()
    print('ECE: CLIP={:.3f}, TTA={:.3f}'.format(clip_ece, tta_ece))
    print('Controls: disagreement={:.4f}, normalized_floor={:.4f}, lr_gate={:.4f}'.format(
        disagreement_meter.avg, floor_meter.avg, lr_gate_meter.avg))

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

    # Disagreement-controlled prompt adaptation (no output temperature scaling).
    parser.add_argument('--agreement_strength', default=1.0, type=float,
                        help='view disagreement weighting strength; 0 uses uniform selected-view weights')
    parser.add_argument('--agreement_temp', default=0.20, type=float,
                        help='softmax temperature for normalized JS view weights')
    parser.add_argument('--entropy_floor_base', default=0.02, type=float,
                        help='base normalized entropy floor')
    parser.add_argument('--entropy_floor_scale', default=0.08, type=float,
                        help='extra normalized entropy floor at maximum sample disagreement')
    parser.add_argument('--sample_lr_min', default=0.25, type=float,
                        help='minimum fraction of the original prompt-tuning LR')
    parser.add_argument('--sample_lr_power', default=1.0, type=float,
                        help='power controlling how disagreement reduces the prompt-tuning LR')
    parser.add_argument('--density_temp', default=0.01, type=float,
                        help='original R-TPT density softmax temperature')
    parser.add_argument('--num_neighbors', default=20, type=int,
                        help='original R-TPT number of feature neighbors')
    parser.add_argument('--ece_bins', default=15, type=int,
                        help='number of confidence bins for ECE')
    parser.add_argument('--eval_both', action='store_true',
                        help='evaluate clean and adversarial metrics in one run; --eps specifies the adversarial budget')

    parser.add_argument('--load_tecoa', type=str, default='', choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'])

    main()