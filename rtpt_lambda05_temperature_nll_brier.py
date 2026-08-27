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


def compute_ece(confidences, corrects, n_bins=15):
    """
    Standard top-label Expected Calibration Error (ECE).
    Returned value is in percentage points.
    """
    confidences = torch.as_tensor(confidences, dtype=torch.float32)
    corrects = torch.as_tensor(corrects, dtype=torch.float32)

    if confidences.numel() == 0:
        return float('nan')

    ece = torch.tensor(0.0, dtype=torch.float32)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)

    for b in range(n_bins):
        lo = boundaries[b]
        hi = boundaries[b + 1]

        if b == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)

        if in_bin.any():
            bin_acc = corrects[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_fraction = in_bin.float().mean()
            ece += torch.abs(bin_acc - bin_conf) * bin_fraction

    return ece.item() * 100.0


def summarize_confidence_metrics(
    confidences,
    corrects,
    n_bins=15,
    high_conf_threshold=0.90,
    nll_values=None,
    brier_values=None
):
    """
    Calibration metrics for one method.

    Acc/ECE/confidence values are reported as percentages.
    NLL is mean negative log-likelihood (natural logarithm).
    Brier is the mean multiclass Brier score:
        sum_c (p_c - 1[c=y])^2
    averaged over samples.

    For a top-label-only diagnostic that has no complete probability
    distribution (e.g. the Hybrid confidence-swap diagnostic), NLL and
    Brier should be left as None and will be reported as NaN/N/A.
    """
    confidences = torch.as_tensor(confidences, dtype=torch.float32)
    corrects = torch.as_tensor(corrects, dtype=torch.bool)

    if confidences.numel() == 0:
        return {
            'Acc': float('nan'),
            'ECE': float('nan'),
            'AvgConf': float('nan'),
            'WrongConf': float('nan'),
            'CorrectConf': float('nan'),
            'HighConfidenceWrong_AllSamples': float('nan'),
            'HighConfidenceWrong_AmongWrong': float('nan'),
            'NLL': float('nan'),
            'Brier': float('nan'),
        }

    wrongs = ~corrects
    high_conf_wrong = wrongs & (confidences >= high_conf_threshold)

    if nll_values is None:
        nll = float('nan')
    else:
        nll_tensor = torch.as_tensor(nll_values, dtype=torch.float32)
        nll = nll_tensor.mean().item() if nll_tensor.numel() > 0 else float('nan')

    if brier_values is None:
        brier = float('nan')
    else:
        brier_tensor = torch.as_tensor(brier_values, dtype=torch.float32)
        brier = brier_tensor.mean().item() if brier_tensor.numel() > 0 else float('nan')

    return {
        'Acc': corrects.float().mean().item() * 100.0,
        'ECE': compute_ece(confidences, corrects.float(), n_bins=n_bins),
        'AvgConf': confidences.mean().item() * 100.0,
        'WrongConf': (
            confidences[wrongs].mean().item() * 100.0
            if wrongs.any() else float('nan')
        ),
        'CorrectConf': (
            confidences[corrects].mean().item() * 100.0
            if corrects.any() else float('nan')
        ),
        'HighConfidenceWrong_AllSamples': (
            high_conf_wrong.float().mean().item() * 100.0
        ),
        'HighConfidenceWrong_AmongWrong': (
            high_conf_wrong.float().sum().item()
            / wrongs.float().sum().item() * 100.0
            if wrongs.any() else float('nan')
        ),
        'NLL': nll,
        'Brier': brier,
    }


def format_metrics_block(title, stats):
    nll_text = (
        "N/A (no full probability distribution)"
        if np.isnan(stats['NLL'])
        else f"{stats['NLL']:.6f}"
    )
    brier_text = (
        "N/A (no full probability distribution)"
        if np.isnan(stats['Brier'])
        else f"{stats['Brier']:.6f}"
    )

    return (
        f"========== {title} ==========\n"
        f"Acc (%): {stats['Acc']:.4f}\n"
        f"ECE (%): {stats['ECE']:.4f}\n"
        f"NLL (mean negative log-likelihood): {nll_text}\n"
        f"Brier (mean multiclass sum over classes): {brier_text}\n"
        f"AvgConf (%): {stats['AvgConf']:.4f}\n"
        f"WrongConf (%): {stats['WrongConf']:.4f}\n"
        f"CorrectConf (%): {stats['CorrectConf']:.4f}\n"
        f"High-confidence wrong (all samples): "
        f"{stats['HighConfidenceWrong_AllSamples']:.4f}%\n"
        f"High-confidence among wrong samples: "
        f"{stats['HighConfidenceWrong_AmongWrong']:.4f}%"
    )


def find_temperature_for_target_confidence(
    logits,
    target_conf,
    max_temperature=100.0,
    num_iterations=40
):
    """
    Find per-sample T >= 1 such that:
        max softmax(logits / T) ~= target_conf

    logits:      [B, C]
    target_conf: [B], target top-1 confidence

    This function only FLATTENS distributions (T >= 1).
    It never sharpens them.
    """
    logits = logits.detach()
    target_conf = target_conf.detach()

    original_prob = torch.softmax(logits, dim=-1)
    original_conf, _ = original_prob.max(dim=-1)

    # If target >= original confidence, no flattening is needed.
    needs_flatten = target_conf < original_conf

    low = torch.ones_like(target_conf)
    high = torch.full_like(target_conf, max_temperature)

    for _ in range(num_iterations):
        mid = (low + high) / 2.0
        mid_prob = torch.softmax(logits / mid.unsqueeze(-1), dim=-1)
        mid_conf, _ = mid_prob.max(dim=-1)

        # T too small -> confidence still above target -> increase T.
        too_confident = mid_conf > target_conf
        low = torch.where(too_confident, mid, low)
        high = torch.where(too_confident, high, mid)

    temperature = (low + high) / 2.0
    temperature = torch.where(needs_flatten, temperature, torch.ones_like(temperature))
    return temperature


def test_time_tuning(model, inputs, optimizer, scaler, args):
    
    selected_idx = None
    for j in range(args.tta_steps):
        if True:
            output = model(inputs) 

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)

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
            val_loader, model, model_state, optimizer, optim_state,
            scaler, args, data_transform
        )
        del val_dataset, val_loader

        report = "\n\n".join([
            format_metrics_block(
                "CLIP Robust / TTA Step 0 Baseline",
                results['clip']
            ),
            format_metrics_block(
                "Prompt Tuning + Uniform Ensemble Robust / TTA Step 1",
                results['uniform']
            ),
            format_metrics_block(
                "R-TPT Robust / TTA Step 1",
                results['rtpt']
            ),
            format_metrics_block(
                "Hybrid Diagnostic: R-TPT Prediction + Uniform Confidence on Agreement",
                results['hybrid']
            ),
            format_metrics_block(
                "Lambda=0.5 Sample-wise Temperature Scaling",
                results['lambda05']
            ),
            (
                "========== Hybrid Diagnostic Summary ==========\n"
                f"Uniform/R-TPT Prediction Agreement Rate: "
                f"{results['agreement_rate']:.4f}%\n"
                f"Hybrid uses Uniform confidence on agreement: "
                f"{results['agreement_rate']:.4f}% of samples\n"
                f"Hybrid keeps R-TPT confidence on disagreement: "
                f"{100.0 - results['agreement_rate']:.4f}% of samples\n"
                f"Hybrid Acc - R-TPT Acc: "
                f"{results['hybrid']['Acc'] - results['rtpt']['Acc']:+.4f}\n"
                f"Hybrid ECE - R-TPT ECE: "
                f"{results['hybrid']['ECE'] - results['rtpt']['ECE']:+.4f}\n"
                f"Hybrid WrongConf - R-TPT WrongConf: "
                f"{results['hybrid']['WrongConf'] - results['rtpt']['WrongConf']:+.4f}\n"
                f"Hybrid High-confidence wrong - R-TPT: "
                f"{results['hybrid']['HighConfidenceWrong_AllSamples'] - results['rtpt']['HighConfidenceWrong_AllSamples']:+.4f}"
            ),
            (
                "========== Lambda=0.5 Temperature Summary ==========\n"
                f"Scaled Sample Rate: {results['lambda05_scaled_rate']:.4f}%\n"
                f"Temperature Mean: {results['lambda05_temperature_mean']:.6f}\n"
                f"Temperature Median: {results['lambda05_temperature_median']:.6f}\n"
                f"Temperature Min: {results['lambda05_temperature_min']:.6f}\n"
                f"Temperature Max: {results['lambda05_temperature_max']:.6f}\n"
                f"Lambda=0.5 Acc - R-TPT Acc: "
                f"{results['lambda05']['Acc'] - results['rtpt']['Acc']:+.4f}\n"
                f"Lambda=0.5 ECE - R-TPT ECE: "
                f"{results['lambda05']['ECE'] - results['rtpt']['ECE']:+.4f}\n"
                f"Lambda=0.5 NLL - R-TPT NLL: "
                f"{results['lambda05']['NLL'] - results['rtpt']['NLL']:+.6f}\n"
                f"Lambda=0.5 Brier - R-TPT Brier: "
                f"{results['lambda05']['Brier'] - results['rtpt']['Brier']:+.6f}\n"
                f"Lambda=0.5 WrongConf - R-TPT WrongConf: "
                f"{results['lambda05']['WrongConf'] - results['rtpt']['WrongConf']:+.4f}\n"
                f"Lambda=0.5 High-confidence wrong - R-TPT: "
                f"{results['lambda05']['HighConfidenceWrong_AllSamples'] - results['rtpt']['HighConfidenceWrong_AllSamples']:+.4f}"
            )
        ])

        args.out_file.write(report + '\n')
        args.out_file.flush()
        print(report + '\n')

        torch.save(results, os.path.join(args.output_dir, 'results_log.pt'))


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    clip_top1 = AverageMeter('CLIP Acc@1', ':6.2f', Summary.AVERAGE)
    uniform_top1 = AverageMeter('Uniform Acc@1', ':6.2f', Summary.AVERAGE)
    rtpt_top1 = AverageMeter('R-TPT Acc@1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, clip_top1, uniform_top1, rtpt_top1],
        prefix='Test: ')

    # Per-sample records used to compute calibration metrics.
    clip_conf_all, clip_correct_all = [], []
    uniform_conf_all, uniform_correct_all = [], []
    rtpt_conf_all, rtpt_correct_all = [], []
    hybrid_conf_all, hybrid_correct_all = [], []
    lambda05_conf_all, lambda05_correct_all = [], []

    # Full-distribution proper scoring rules.
    clip_nll_all, clip_brier_all = [], []
    uniform_nll_all, uniform_brier_all = [], []
    rtpt_nll_all, rtpt_brier_all = [], []
    lambda05_nll_all, lambda05_brier_all = [], []
    lambda05_temperature_all = []
    lambda05_scaled_all = []
    agreement_all = []

    # reset model and switch to evaluate mode
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

        # Keep attack generation identical to the supplied baseline code.
        # This modification only changes evaluation/metrics.
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
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images

        images = torch.cat(images, dim=0)

        # reset model
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)

        # Step-0 CLIP + untuned image features for R-TPT reliability.
        with torch.no_grad():
            clip_output = model(image)
            clip_features, _, _ = model.forward_features(images)

        # Test-time prompt tuning.
        assert args.tta_steps > 0
        test_time_tuning(model, images, optimizer, scaler, args)

        with torch.no_grad():
            # Shape: [num_views, num_classes]
            tuned_outputs = model(images)

        # ------------------------------------------------------------
        # 1) Prompt Tuning + Uniform Ensemble
        # ------------------------------------------------------------
        uniform_output = tuned_outputs.mean(dim=0, keepdim=True)

        # ------------------------------------------------------------
        # 2) R-TPT reliability-weighted ensemble
        # ------------------------------------------------------------
        sim_matrix_images = torch.bmm(
            clip_features.unsqueeze(0),
            clip_features.unsqueeze(0).permute(0, 2, 1)
        )
        score = get_top_sim(sim_matrix_images)
        weight = torch.nn.functional.softmax(score / 0.01, dim=-1)
        rtpt_output = torch.bmm(
            weight.unsqueeze(-1).transpose(1, 2),
            tuned_outputs.unsqueeze(0)
        ).squeeze(1)

        # Top-label prediction/confidence.
        clip_prob = torch.softmax(clip_output, dim=-1)
        uniform_prob = torch.softmax(uniform_output, dim=-1)
        rtpt_prob = torch.softmax(rtpt_output, dim=-1)

        clip_conf, clip_pred = clip_prob.max(dim=-1)
        uniform_conf, uniform_pred = uniform_prob.max(dim=-1)
        rtpt_conf, rtpt_pred = rtpt_prob.max(dim=-1)

        clip_correct = clip_pred.eq(target)
        uniform_correct = uniform_pred.eq(target)
        rtpt_correct = rtpt_pred.eq(target)

        # ------------------------------------------------------------
        # 3) Hybrid diagnostic
        #
        # Prediction is ALWAYS R-TPT's prediction.
        #
        # If Uniform and R-TPT predict the same class:
        #     confidence = Uniform confidence
        # Else:
        #     confidence = R-TPT confidence
        #
        # Therefore Hybrid accuracy is exactly R-TPT accuracy.
        # This is a top-label confidence diagnostic, not yet a full
        # probability-distribution calibration method.
        # ------------------------------------------------------------
        agree = uniform_pred.eq(rtpt_pred)
        hybrid_conf = torch.where(agree, uniform_conf, rtpt_conf)
        hybrid_correct = rtpt_correct

        # ------------------------------------------------------------
        # 4) Lambda=0.5 sample-wise temperature scaling
        #
        # We only use Uniform confidence as a reference when Uniform and
        # R-TPT predict the SAME top-1 class.
        #
        # On agreement samples:
        #   if R-TPT is more confident than Uniform:
        #       C_target = C_U + 0.5 * (C_R - C_U)
        #   else:
        #       keep C_target = C_R  (no sharpening)
        #
        # On disagreement samples:
        #   keep C_target = C_R      (do not use Uniform confidence
        #                              from a different predicted class)
        #
        # Then find T(x) >= 1 such that:
        #   max softmax(z_R / T(x)) ~= C_target
        #
        # Because T(x) is positive and applied uniformly across classes,
        # the R-TPT top-1 prediction is unchanged, so accuracy is preserved.
        # ------------------------------------------------------------
        lambda05_target_conf = rtpt_conf.clone()

        can_flatten = agree & (rtpt_conf > uniform_conf)
        midpoint_conf = uniform_conf + 0.1 * (rtpt_conf - uniform_conf)
        lambda05_target_conf = torch.where(
            can_flatten,
            midpoint_conf,
            lambda05_target_conf
        )

        lambda05_temperature = find_temperature_for_target_confidence(
            rtpt_output,
            lambda05_target_conf
        )
        lambda05_output = rtpt_output / lambda05_temperature.unsqueeze(-1)
        lambda05_prob = torch.softmax(lambda05_output, dim=-1)
        lambda05_conf, lambda05_pred = lambda05_prob.max(dim=-1)
        lambda05_correct = lambda05_pred.eq(target)

        # ------------------------------------------------------------
        # Proper scoring rules: NLL and multiclass Brier score.
        #
        # NLL:
        #   -log p(y_true)
        #
        # Brier:
        #   sum_c (p_c - 1[c=y_true])^2
        #
        # These require a COMPLETE probability distribution, therefore
        # they are computed for CLIP / Uniform / R-TPT / Lambda=0.5 TS.
        # They are NOT defined for the top-label-only Hybrid diagnostic.
        # ------------------------------------------------------------
        clip_nll = torch.nn.functional.cross_entropy(
            clip_output, target, reduction='none'
        )
        uniform_nll = torch.nn.functional.cross_entropy(
            uniform_output, target, reduction='none'
        )
        rtpt_nll = torch.nn.functional.cross_entropy(
            rtpt_output, target, reduction='none'
        )
        lambda05_nll = torch.nn.functional.cross_entropy(
            lambda05_output, target, reduction='none'
        )

        one_hot_target = torch.nn.functional.one_hot(
            target,
            num_classes=rtpt_prob.size(-1)
        ).to(dtype=rtpt_prob.dtype)

        clip_brier = ((clip_prob - one_hot_target) ** 2).sum(dim=-1)
        uniform_brier = ((uniform_prob - one_hot_target) ** 2).sum(dim=-1)
        rtpt_brier = ((rtpt_prob - one_hot_target) ** 2).sum(dim=-1)
        lambda05_brier = ((lambda05_prob - one_hot_target) ** 2).sum(dim=-1)

        # Safety check: temperature scaling must preserve R-TPT top-1.
        if not torch.equal(lambda05_pred, rtpt_pred):
            raise RuntimeError(
                "Lambda=0.5 temperature scaling changed the R-TPT top-1 prediction."
            )

        # Running accuracy meters.
        clip_acc1, _ = accuracy(clip_output, target, topk=(1, 5))
        uniform_acc1, _ = accuracy(uniform_output, target, topk=(1, 5))
        rtpt_acc1, _ = accuracy(rtpt_output, target, topk=(1, 5))

        sample_count = target.size(0)
        clip_top1.update(clip_acc1[0], sample_count)
        uniform_top1.update(uniform_acc1[0], sample_count)
        rtpt_top1.update(rtpt_acc1[0], sample_count)

        # Store per-sample information on CPU.
        clip_conf_all.extend(clip_conf.detach().cpu().tolist())
        clip_correct_all.extend(clip_correct.detach().cpu().tolist())

        uniform_conf_all.extend(uniform_conf.detach().cpu().tolist())
        uniform_correct_all.extend(uniform_correct.detach().cpu().tolist())

        rtpt_conf_all.extend(rtpt_conf.detach().cpu().tolist())
        rtpt_correct_all.extend(rtpt_correct.detach().cpu().tolist())

        hybrid_conf_all.extend(hybrid_conf.detach().cpu().tolist())
        hybrid_correct_all.extend(hybrid_correct.detach().cpu().tolist())

        lambda05_conf_all.extend(lambda05_conf.detach().cpu().tolist())
        lambda05_correct_all.extend(lambda05_correct.detach().cpu().tolist())

        clip_nll_all.extend(clip_nll.detach().cpu().tolist())
        clip_brier_all.extend(clip_brier.detach().cpu().tolist())
        uniform_nll_all.extend(uniform_nll.detach().cpu().tolist())
        uniform_brier_all.extend(uniform_brier.detach().cpu().tolist())
        rtpt_nll_all.extend(rtpt_nll.detach().cpu().tolist())
        rtpt_brier_all.extend(rtpt_brier.detach().cpu().tolist())
        lambda05_nll_all.extend(lambda05_nll.detach().cpu().tolist())
        lambda05_brier_all.extend(lambda05_brier.detach().cpu().tolist())

        lambda05_temperature_all.extend(lambda05_temperature.detach().cpu().tolist())
        lambda05_scaled_all.extend(can_flatten.detach().cpu().tolist())

        agreement_all.extend(agree.detach().cpu().tolist())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(val_loader):
            if args.eps <= 0:
                print_log = (
                    'iter:{}/{}, clip_acc1={}, uniform_acc1={}, rtpt_acc1={}'
                    .format(
                        i, len(val_loader),
                        clip_top1.avg, uniform_top1.avg, rtpt_top1.avg
                    )
                )
            else:
                print_log = (
                    'iter:{}/{}, clip_adv1={}, uniform_adv1={}, rtpt_adv1={}'
                    .format(
                        i, len(val_loader),
                        clip_top1.avg, uniform_top1.avg, rtpt_top1.avg
                    )
                )

            args.out_file.write(print_log + '\n')
            args.out_file.flush()
            print(print_log + '\n')
            progress.display(i)

    progress.display_summary()

    clip_stats = summarize_confidence_metrics(
        clip_conf_all,
        clip_correct_all,
        n_bins=args.ece_bins,
        high_conf_threshold=args.high_conf_threshold,
        nll_values=clip_nll_all,
        brier_values=clip_brier_all
    )
    uniform_stats = summarize_confidence_metrics(
        uniform_conf_all,
        uniform_correct_all,
        n_bins=args.ece_bins,
        high_conf_threshold=args.high_conf_threshold,
        nll_values=uniform_nll_all,
        brier_values=uniform_brier_all
    )
    rtpt_stats = summarize_confidence_metrics(
        rtpt_conf_all,
        rtpt_correct_all,
        n_bins=args.ece_bins,
        high_conf_threshold=args.high_conf_threshold,
        nll_values=rtpt_nll_all,
        brier_values=rtpt_brier_all
    )
    hybrid_stats = summarize_confidence_metrics(
        hybrid_conf_all,
        hybrid_correct_all,
        n_bins=args.ece_bins,
        high_conf_threshold=args.high_conf_threshold,
        nll_values=None,
        brier_values=None
    )
    lambda05_stats = summarize_confidence_metrics(
        lambda05_conf_all,
        lambda05_correct_all,
        n_bins=args.ece_bins,
        high_conf_threshold=args.high_conf_threshold,
        nll_values=lambda05_nll_all,
        brier_values=lambda05_brier_all
    )

    agreement_rate = (
        float(np.mean(agreement_all) * 100.0)
        if agreement_all else float('nan')
    )

    lambda05_scaled_rate = (
        float(np.mean(lambda05_scaled_all) * 100.0)
        if lambda05_scaled_all else float('nan')
    )
    lambda05_temperature_np = np.asarray(lambda05_temperature_all, dtype=np.float64)
    if lambda05_temperature_np.size > 0:
        lambda05_temperature_mean = float(lambda05_temperature_np.mean())
        lambda05_temperature_median = float(np.median(lambda05_temperature_np))
        lambda05_temperature_min = float(lambda05_temperature_np.min())
        lambda05_temperature_max = float(lambda05_temperature_np.max())
    else:
        lambda05_temperature_mean = float('nan')
        lambda05_temperature_median = float('nan')
        lambda05_temperature_min = float('nan')
        lambda05_temperature_max = float('nan')

    return {
        'clip': clip_stats,
        'uniform': uniform_stats,
        'rtpt': rtpt_stats,
        'hybrid': hybrid_stats,
        'lambda05': lambda05_stats,
        'agreement_rate': agreement_rate,
        'lambda05_scaled_rate': lambda05_scaled_rate,
        'lambda05_temperature_mean': lambda05_temperature_mean,
        'lambda05_temperature_median': lambda05_temperature_median,
        'lambda05_temperature_min': lambda05_temperature_min,
        'lambda05_temperature_max': lambda05_temperature_max,
        'hybrid_definition': (
            'Prediction always follows R-TPT. Confidence uses Uniform when '
            'Uniform and R-TPT top-1 predictions agree; otherwise confidence '
            'uses R-TPT.'
        ),
        'lambda05_definition': (
            'On Uniform/R-TPT agreement samples where R-TPT confidence is '
            'higher, target confidence is the midpoint between Uniform and '
            'R-TPT confidence (lambda=0.5), implemented by per-sample '
            'temperature scaling with T>=1. On disagreement or when R-TPT is '
            'not more confident, T=1.'
        ),
        'ece_bins': args.ece_bins,
        'high_conf_threshold': args.high_conf_threshold,
    }


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

    # Calibration metric settings
    parser.add_argument(
        '--ece_bins',
        default=15,
        type=int,
        help='number of bins used for top-label ECE'
    )
    parser.add_argument(
        '--high_conf_threshold',
        default=0.90,
        type=float,
        help='confidence threshold used to define high-confidence wrong predictions'
    )

    parser.add_argument('--load_tecoa', type=str, default='', choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'])

    main()