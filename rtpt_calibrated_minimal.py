import argparse
import math
import os
import time
from copy import deepcopy

from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torchattacks

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import *
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def entropy_per_view(logits):
    """Return the original softmax pointwise entropy for every view."""
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def entropy_avg(outputs):
    return entropy_per_view(outputs).mean()


def select_confident_samples(logits, top):
    """Original R-TPT low-entropy view selection."""
    batch_entropy = entropy_per_view(logits)
    num_keep = max(1, int(batch_entropy.size(0) * top))
    idx = torch.argsort(batch_entropy, descending=False)[:num_keep]
    return logits[idx], idx


@torch.no_grad()
def shared_random_prompt_sensitivity(model, candidate_inputs, clean_entropy, rho=1e-3):
    """
    Give every low-entropy candidate view the same temporary random prompt
    perturbation and measure its absolute entropy change.

    This is a random-direction prompt sensitivity proxy, not exact SAM
    sharpness. The perturbation is restored immediately and never becomes
    part of the real prompt update.
    """
    prompt_params = [
        p for p in model.prompt_learner.parameters()
        if p.requires_grad
    ]
    if not prompt_params:
        raise RuntimeError("No trainable prompt parameters were found.")

    noises = [torch.randn_like(p) for p in prompt_params]
    noise_norm_sq = torch.zeros((), device=noises[0].device, dtype=torch.float32)
    for noise in noises:
        noise_norm_sq += noise.float().pow(2).sum()
    noise_norm = noise_norm_sq.sqrt().clamp_min(1e-12)

    perturbations = []
    try:
        for param, noise in zip(prompt_params, noises):
            perturb = (rho * noise.float() / noise_norm).to(param.dtype)
            param.add_(perturb)
            perturbations.append(perturb)

        perturbed_logits = model(candidate_inputs)
        perturbed_entropy = entropy_per_view(perturbed_logits)
    finally:
        for param, perturb in zip(prompt_params, perturbations):
            param.sub_(perturb)

    return (perturbed_entropy - clean_entropy).abs()


def test_time_tuning(model, inputs, optimizer, scaler, args):
    """
    Sensitivity-only view filtering:
      1. Measure shared-random prompt sensitivity on all augmented views.
      2. Drop the highest-sensitivity 20% of all views (configurable).
      3. Optimize the original pointwise entropy on every remaining view.

    Example with 64 views and drop_ratio=0.2:
      64 -> drop ceil(64*0.2)=13 -> keep 51 views -> entropy update.

    Note: this removes R-TPT's original low-entropy top-p view selection.
    """
    del scaler

    selected_idx = None
    sensitivity_stats = {}

    for _ in range(args.tta_steps):
        if selected_idx is None:
            with torch.no_grad():
                clean_logits = model(inputs)
                clean_entropy = entropy_per_view(clean_logits)

            sensitivity = shared_random_prompt_sensitivity(
                model=model,
                candidate_inputs=inputs,
                clean_entropy=clean_entropy,
                rho=args.sensitivity_rho,
            )

            total_views = inputs.size(0)
            if args.sensitivity_drop_ratio > 0.0:
                num_drop = min(
                    total_views - 1,
                    math.ceil(total_views * args.sensitivity_drop_ratio),
                )
            else:
                num_drop = 0

            num_keep = total_views - num_drop
            selected_idx = torch.topk(
                sensitivity,
                k=num_keep,
                largest=False,
            ).indices

            sensitivity_stats = {
                "total_view_count": int(total_views),
                "sensitivity_dropped_count": int(num_drop),
                "selected_count": int(selected_idx.numel()),
                "sensitivity_mean": float(sensitivity.mean().item()),
                "sensitivity_max": float(sensitivity.max().item()),
                "sensitivity_min": float(sensitivity.min().item()),
            }

        selected_logits = model(inputs[selected_idx])
        loss = entropy_avg(selected_logits)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return sensitivity_stats


def get_top_sim(sim_matrix, k=20):
    """Mean similarity to the top-k other augmented views."""
    if sim_matrix.dim() != 3 or sim_matrix.size(-1) != sim_matrix.size(-2):
        raise ValueError("sim_matrix must have shape [B, N, N].")

    num_views = sim_matrix.size(-1)
    if num_views <= 1:
        return torch.ones(
            sim_matrix.size(0),
            num_views,
            device=sim_matrix.device,
            dtype=sim_matrix.dtype,
        )

    k = min(k, num_views - 1)
    sim_matrix = sim_matrix.clone()
    diagonal = torch.eye(
        num_views,
        device=sim_matrix.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    sim_matrix.masked_fill_(diagonal, float("-inf"))

    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    return top_k_values.mean(dim=-1)


@torch.no_grad()
def reliability_weighted_prediction(model, images, image_features, temperature=0.01):
    """Original R-TPT reliability-weighted multi-view ensemble."""
    tuned_outputs = model(images)
    sim_matrix = torch.bmm(
        image_features.unsqueeze(0),
        image_features.unsqueeze(0).transpose(1, 2),
    )
    score = get_top_sim(sim_matrix)
    weight = F.softmax(score / temperature, dim=-1)
    return torch.bmm(
        weight.unsqueeze(-1).transpose(1, 2),
        tuned_outputs.unsqueeze(0),
    ).squeeze(1)


def reset_model_and_optimizer(model, optimizer, optim_state):
    with torch.no_grad():
        model.reset()
    optimizer.load_state_dict(optim_state)
    optimizer.zero_grad(set_to_none=True)


def loader_views_to_tensor(images, gpu):
    """Convert loader output to one [N,C,H,W] CUDA tensor."""
    if isinstance(images, list):
        cuda_views = [view.cuda(gpu, non_blocking=True) for view in images]
        return torch.cat(cuda_views, dim=0)

    if images.dim() > 4:
        if images.size(0) != 1:
            raise ValueError("Expected outer batch size 1 for multi-view input.")
        images = images.squeeze(0)

    return images.cuda(gpu, non_blocking=True)


def adversarial_tensor_to_views(adversarial_image, data_transform, gpu):
    """Turn one adversarial tensor into the same AugMix view batch."""
    adversarial_pil = transforms.ToPILImage()(
        adversarial_image.detach().cpu().squeeze(0)
    )
    views = data_transform(adversarial_pil)
    if not isinstance(views, list):
        views = [views]

    batched_views = []
    for view in views:
        if view.dim() == 3:
            view = view.unsqueeze(0)
        batched_views.append(view.cuda(gpu, non_blocking=True))
    return torch.cat(batched_views, dim=0)


def append_calibration_data(logits, target, confidence_store, correct_store):
    probabilities = logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(target)

    confidence_store.extend(confidence.detach().float().cpu().tolist())
    correct_store.extend(correct.detach().float().cpu().tolist())


def expected_calibration_error(confidences, correctness, n_bins=15):
    """
    Equal-width ECE on a 0-100 percentage scale, matching reported accuracy.
    """
    if len(confidences) == 0:
        return float("nan")

    confidences = torch.tensor(confidences, dtype=torch.float32)
    correctness = torch.tensor(correctness, dtype=torch.float32)
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.zeros((), dtype=torch.float32)

    for bin_idx in range(n_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        if in_bin.any():
            bin_weight = in_bin.float().mean()
            bin_accuracy = correctness[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += bin_weight * (bin_accuracy - bin_confidence).abs()

    return float((ece * 100.0).item())


def adapt_and_predict(model, images, optimizer, optim_state, scaler, args):
    """Reset, adapt once, and return original and adapted predictions."""
    reset_model_and_optimizer(model, optimizer, optim_state)

    with torch.no_grad():
        original_output = model(images[:1])
        image_features, _, _ = model.forward_features(images)

    sensitivity_stats = test_time_tuning(
        model,
        images,
        optimizer,
        scaler,
        args,
    )

    with torch.no_grad():
        adapted_output = reliability_weighted_prediction(
            model=model,
            images=images,
            image_features=image_features,
            temperature=args.reliability_temperature,
        )

    return original_output, adapted_output, sensitivity_stats


def test_time_adapt_eval(
    val_loader,
    model,
    model_state,
    optimizer,
    optim_state,
    scaler,
    args,
    data_transform,
):
    del model_state

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    clip_clean_acc = AverageMeter("CLIP-Clean@1", ":6.2f", Summary.AVERAGE)
    method_clean_acc = AverageMeter("Ours-Clean@1", ":6.2f", Summary.AVERAGE)
    clip_robust_acc = AverageMeter("CLIP-Robust@1", ":6.2f", Summary.AVERAGE)
    method_robust_acc = AverageMeter("Ours-Robust@1", ":6.2f", Summary.AVERAGE)
    clean_sensitivity = AverageMeter("Clean-Sens", ":8.5f", Summary.AVERAGE)
    robust_sensitivity = AverageMeter("Robust-Sens", ":8.5f", Summary.AVERAGE)

    robust_enabled = args.eps > 0.0 and args.steps > 0
    progress_meters = [batch_time, clip_clean_acc, method_clean_acc]
    if robust_enabled:
        progress_meters.extend([clip_robust_acc, method_robust_acc])

    progress = ProgressMeter(len(val_loader), progress_meters, prefix="Test: ")

    calibration = {
        "clip_clean_conf": [],
        "clip_clean_correct": [],
        "method_clean_conf": [],
        "method_clean_correct": [],
        "clip_robust_conf": [],
        "clip_robust_correct": [],
        "method_robust_conf": [],
        "method_robust_correct": [],
    }

    model.eval()
    attack = None
    if robust_enabled:
        attack = torchattacks.PGD(
            model,
            eps=args.eps / 255.0,
            alpha=args.alpha / 255.0,
            steps=args.steps,
        )

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        target = target.cuda(args.gpu, non_blocking=True)
        clean_views = loader_views_to_tensor(images, args.gpu)

        # Clean evaluation.
        clean_clip_output, clean_method_output, clean_stats = adapt_and_predict(
            model=model,
            images=clean_views,
            optimizer=optimizer,
            optim_state=optim_state,
            scaler=scaler,
            args=args,
        )

        clean_clip_top1, _ = accuracy(clean_clip_output, target, topk=(1, 5))
        clean_method_top1, _ = accuracy(clean_method_output, target, topk=(1, 5))
        clip_clean_acc.update(clean_clip_top1[0], target.size(0))
        method_clean_acc.update(clean_method_top1[0], target.size(0))
        clean_sensitivity.update(
            clean_stats.get("sensitivity_mean", 0.0),
            target.size(0),
        )

        append_calibration_data(
            clean_clip_output,
            target,
            calibration["clip_clean_conf"],
            calibration["clip_clean_correct"],
        )
        append_calibration_data(
            clean_method_output,
            target,
            calibration["method_clean_conf"],
            calibration["method_clean_correct"],
        )

        # Robust evaluation. The attack is generated from the reset/original
        # prompt, not from the prompt adapted on the clean image above.
        if robust_enabled:
            reset_model_and_optimizer(model, optimizer, optim_state)
            adversarial_image = attack(clean_views[:1], target)
            model.zero_grad(set_to_none=True)

            adversarial_views = adversarial_tensor_to_views(
                adversarial_image=adversarial_image,
                data_transform=data_transform,
                gpu=args.gpu,
            )

            robust_clip_output, robust_method_output, robust_stats = adapt_and_predict(
                model=model,
                images=adversarial_views,
                optimizer=optimizer,
                optim_state=optim_state,
                scaler=scaler,
                args=args,
            )

            robust_clip_top1, _ = accuracy(robust_clip_output, target, topk=(1, 5))
            robust_method_top1, _ = accuracy(robust_method_output, target, topk=(1, 5))
            clip_robust_acc.update(robust_clip_top1[0], target.size(0))
            method_robust_acc.update(robust_method_top1[0], target.size(0))
            robust_sensitivity.update(
                robust_stats.get("sensitivity_mean", 0.0),
                target.size(0),
            )

            append_calibration_data(
                robust_clip_output,
                target,
                calibration["clip_robust_conf"],
                calibration["clip_robust_correct"],
            )
            append_calibration_data(
                robust_method_output,
                target,
                calibration["method_robust_conf"],
                calibration["method_robust_correct"],
            )

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(val_loader):
            print_log = (
                "iter:{}/{}, clip_clean_acc={}, ours_clean_acc={}, "
                "clean_sensitivity={:.6f}"
            ).format(
                i + 1,
                len(val_loader),
                clip_clean_acc.avg,
                method_clean_acc.avg,
                clean_sensitivity.avg,
            )
            if robust_enabled:
                print_log += (
                    ", clip_robust_acc={}, ours_robust_acc={}, "
                    "robust_sensitivity={:.6f}"
                ).format(
                    clip_robust_acc.avg,
                    method_robust_acc.avg,
                    robust_sensitivity.avg,
                )

            args.out_file.write(print_log + "\n")
            args.out_file.flush()
            print(print_log + "\n")
            progress.display(i + 1)

    progress.display_summary()

    metrics = {
        "clip_clean_acc": float(clip_clean_acc.avg),
        "clip_clean_ece": expected_calibration_error(
            calibration["clip_clean_conf"],
            calibration["clip_clean_correct"],
            n_bins=args.ece_bins,
        ),
        "method_clean_acc": float(method_clean_acc.avg),
        "method_clean_ece": expected_calibration_error(
            calibration["method_clean_conf"],
            calibration["method_clean_correct"],
            n_bins=args.ece_bins,
        ),
        "clean_sensitivity_mean": float(clean_sensitivity.avg),
    }

    if robust_enabled:
        metrics.update({
            "clip_robust_acc": float(clip_robust_acc.avg),
            "clip_robust_ece": expected_calibration_error(
                calibration["clip_robust_conf"],
                calibration["clip_robust_correct"],
                n_bins=args.ece_bins,
            ),
            "method_robust_acc": float(method_robust_acc.avg),
            "method_robust_ece": expected_calibration_error(
                calibration["method_robust_conf"],
                calibration["method_robust_correct"],
                n_bins=args.ece_bins,
            ),
            "robust_sensitivity_mean": float(robust_sensitivity.avg),
        })

    return metrics


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    if not 0.0 < args.selection_p <= 1.0:
        raise ValueError("--selection_p must be in (0, 1].")
    if not 0.0 <= args.sensitivity_drop_ratio < 1.0:
        raise ValueError("--sensitivity_drop_ratio must be in [0, 1).")
    if args.sensitivity_rho < 0.0:
        raise ValueError("--sensitivity_rho must be non-negative.")
    if args.eps > 0.0 and args.steps <= 0:
        raise ValueError("Robust evaluation requires --steps > 0 when --eps > 0.")

    args.alpha = args.eps / 4.0
    args.output_dir = os.path.join(
        args.output_dir,
        args.arch,
        args.test_sets,
        "eps_{}_alpha_{}_step_{}".format(args.eps, args.alpha, args.steps),
    )
    os.makedirs(args.output_dir, exist_ok=True)

    args.out_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()

    if args.gpu is None:
        raise ValueError("--gpu must be specified.")

    print("Use GPU: {} for evaluation".format(args.gpu))

    dset = args.test_sets
    if len(dset) > 1:
        classnames = eval("{}_classes".format(dset.lower()))
    else:
        if dset not in ["A", "R", "K", "V", "I"]:
            raise ValueError("Unknown ImageNet variant: {}".format(dset))

        classnames_all = imagenet_classes
        classnames = []
        if dset in ["A", "R", "V"]:
            label_mask = eval("imagenet_{}_mask".format(dset.lower()))
            if dset == "R":
                for class_idx, enabled in enumerate(label_mask):
                    if enabled:
                        classnames.append(classnames_all[class_idx])
            else:
                classnames = [classnames_all[class_idx] for class_idx in label_mask]
        else:
            classnames = classnames_all

    args.classnames = classnames

    model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
    model_state = None

    if len(args.load_tecoa) > 0:
        robust_pretrain_paths = {
            "RN50-eps1": "pretrain/tecoa/rn50_eps1.pth.tar",
        }
        if args.load_tecoa not in robust_pretrain_paths:
            raise ValueError(
                "No checkpoint path configured for {}".format(args.load_tecoa)
            )

        args.robust_pretrain_path = robust_pretrain_paths[args.load_tecoa]
        robust_state_dict = torch.load(args.robust_pretrain_path, map_location="cpu")
        model.image_encoder.load_state_dict(
            robust_state_dict["vision_encoder_state_dict"]
        )
        print("Loaded robust vision encoder.")

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this evaluation script.")

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    trainable_params = [
        p for p in model.prompt_learner.parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    optim_state = deepcopy(optimizer.state_dict())

    scaler = None
    cudnn.benchmark = True

    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_transform = AugMixAugmenter(
        base_transform,
        preprocess,
        n_views=args.batch_size - 1,
        augmix=len(dset) > 1,
    )

    val_dataset = build_dataset(
        dset,
        data_transform,
        args.data,
        mode=args.dataset_mode,
    )
    print("Number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Evaluating: {}".format(dset))

    metrics = test_time_adapt_eval(
        val_loader=val_loader,
        model=model,
        model_state=model_state,
        optimizer=optimizer,
        optim_state=optim_state,
        scaler=scaler,
        args=args,
        data_transform=data_transform,
    )

    summary_lines = [
        "=> Dataset [{}]".format(dset),
        "CLIP Clean Acc: {:.4f}".format(metrics["clip_clean_acc"]),
        "CLIP Clean ECE: {:.4f}".format(metrics["clip_clean_ece"]),
        "Ours Clean Acc: {:.4f}".format(metrics["method_clean_acc"]),
        "Ours Clean ECE: {:.4f}".format(metrics["method_clean_ece"]),
    ]

    if "method_robust_acc" in metrics:
        summary_lines.extend([
            "CLIP Robust Acc: {:.4f}".format(metrics["clip_robust_acc"]),
            "CLIP Robust ECE: {:.4f}".format(metrics["clip_robust_ece"]),
            "Ours Robust Acc: {:.4f}".format(metrics["method_robust_acc"]),
            "Ours Robust ECE: {:.4f}".format(metrics["method_robust_ece"]),
        ])
    else:
        summary_lines.append(
            "Robust metrics skipped because --eps <= 0 or --steps <= 0."
        )

    print_log = "\n".join(summary_lines)
    args.out_file.write(print_log + "\n")
    args.out_file.flush()
    print(print_log + "\n")

    torch.save(metrics, os.path.join(args.output_dir, "results_log.pt"))
    args.out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensitivity-filtered Robust Test-time Prompt Tuning"
    )
    parser.add_argument("data", metavar="DIR", help="path to dataset root")
    parser.add_argument("--test_sets", type=str, default="Caltech101")
    parser.add_argument("--dataset_mode", type=str, default="test")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="RN50")
    parser.add_argument("--resolution", default=224, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N")
    parser.add_argument("-b", "--batch-size", default=64, type=int, metavar="N")
    parser.add_argument("-p", "--print-freq", default=200, type=int, metavar="N")
    parser.add_argument("--gpu", default=0, type=int)

    parser.add_argument("--n_ctx", default=4, type=int)
    parser.add_argument("--ctx_init", default=None, type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_results/ckps/rtpt_sensitivity",
    )

    parser.add_argument("--eps", default=0.0, type=float)
    parser.add_argument("--alpha", default=0.0, type=float)
    parser.add_argument("--steps", type=int, default=0)

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=5e-3,
        type=float,
        metavar="LR",
        dest="lr",
    )
    parser.add_argument("--selection_p", default=0.1, type=float)
    parser.add_argument("--tta_steps", default=1, type=int)

    parser.add_argument(
        "--sensitivity_rho",
        default=1e-3,
        type=float,
        help="global L2 radius of the shared random prompt perturbation",
    )
    parser.add_argument(
        "--sensitivity_drop_ratio",
        default=0.2,
        type=float,
        help="drop the highest-sensitivity fraction after low-entropy selection",
    )
    parser.add_argument("--ece_bins", default=15, type=int)
    parser.add_argument(
        "--reliability_temperature",
        default=0.01,
        type=float,
    )

    parser.add_argument(
        "--load_tecoa",
        type=str,
        default="",
        choices=["", "RN50-eps1", "ViT-B/32-eps1", "ViT-B/32-eps4"],
    )

    main()