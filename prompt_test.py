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
from utils.tools import (
    Summary,
    AverageMeter,
    ProgressMeter,
    accuracy,
    load_model_weight,
    set_random_seed,
)
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import (
    thousand_k_to_200,
    imagenet_a_mask,
    imagenet_r_mask,
    imagenet_v_mask,
)
import os

import torchattacks


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(models.__dict__[name])
)


def get_top_sim(sim_matrix):
    """
    Original R-TPT reliability score:
    for each augmented view, average similarity to its top-20 neighbours.
    Higher score = more consistent/reliable view.
    """
    k = 20

    if sim_matrix.dim() == 3:
        sim_matrix = sim_matrix.squeeze(0)

    sim_matrix = sim_matrix.clone()
    n = sim_matrix.size(0)

    # Avoid selecting self-similarity.
    eye = torch.eye(n, device=sim_matrix.device, dtype=torch.bool)
    sim_matrix[eye] = float("-inf")

    # Be safe when the number of views is smaller than 21.
    k = min(k, max(n - 1, 1))

    top_k_values, _ = sim_matrix.topk(k, dim=-1)
    top_k_mean = top_k_values.mean(dim=-1)

    return top_k_mean  # [N]


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        # Avoid printing the file handle itself.
        if arg == "out_file":
            continue
        s += "{}:{}\n".format(arg, content)
    return s


def select_confident_samples(logits, top):
    """
    Select the lowest-entropy fraction of augmented views.
    This is the original TPT/R-TPT confidence selection.
    """
    batch_entropy = -(
        logits.softmax(1) * logits.log_softmax(1)
    ).sum(1)

    num_selected = int(batch_entropy.size(0) * top)
    num_selected = max(1, min(num_selected, batch_entropy.size(0)))

    idx = torch.argsort(
        batch_entropy,
        descending=False
    )[:num_selected]

    return logits[idx], idx


def entropy_avg(outputs):
    """
    Original entropy-minimization objective.
    """
    batch_entropy = -(
        outputs.softmax(1) * outputs.log_softmax(1)
    ).sum(1)

    return batch_entropy.mean()

def prediction_agreement(logits):
    """
    logits: [num_views, num_classes]

    返回：
    top1_agreement：多数类别所占比例
    prob_agreement：完整概率分布的一致程度
    """
    probs = logits.softmax(dim=-1)

    # Top-1 voting agreement
    predictions = probs.argmax(dim=-1)
    counts = torch.bincount(
        predictions,
        minlength=probs.size(1)
    )
    top1_agreement = counts.max().float() / predictions.numel()

    # 每个视图与平均预测之间的 JS divergence
    mean_probs = probs.mean(dim=0, keepdim=True)

    eps = 1e-8
    probs_safe = probs.clamp_min(eps)
    mean_safe = mean_probs.expand_as(probs).clamp_min(eps)

    mixture = 0.5 * (probs_safe + mean_safe)

    js = 0.5 * (
        (
            probs_safe
            * (probs_safe.log() - mixture.log())
        ).sum(dim=-1)
        +
        (
            mean_safe
            * (mean_safe.log() - mixture.log())
        ).sum(dim=-1)
    )

    # 越接近 1，预测概率分布越一致
    prob_agreement = (1.0 - js.mean() / np.log(2.0)).clamp(0.0, 1.0)

   
    return {
        "top1_agreement": top1_agreement.item(),
        "prob_agreement": prob_agreement.item(),
        "num_unique_predictions": predictions.unique().numel(),
    }
def flatten_gradients(grads, params):
    flattened = []

    for grad, param in zip(grads, params):
        if grad is None:
            flattened.append(
                torch.zeros_like(param).reshape(-1)
            )
        else:
            flattened.append(
                grad.detach().reshape(-1)
            )

    return torch.cat(flattened)


def prompt_gradient_agreement(
    model,
    inputs,
    selection_p=0.1,
    max_views=6
):
    """
    对低熵视图逐个计算：
        g_i = d H(p_i) / d prompt

    返回各视图 prompt gradient 的方向一致程度。
    """

    prompt_params = [
        param
        for name, param in model.named_parameters()
        if "prompt_learner" in name
        and param.requires_grad
    ]
    if len(prompt_params) == 0:
        raise RuntimeError(
            "No trainable prompt parameters found."
        )
    # 这里不能放在 torch.no_grad() 内
    logits = model(inputs)

    # 与 R-TPT 使用相同的低熵视图
    _, selected_idx = select_confident_samples(
        logits.detach(),
        selection_p
    )

    selected_idx = selected_idx[:max_views]

    grad_vectors = []

    for position, view_idx in enumerate(selected_idx.tolist()):
        view_logits = logits[
            view_idx:view_idx + 1
        ]

        view_loss = entropy_avg(view_logits)

        grads = torch.autograd.grad(
            view_loss,
            prompt_params,
            retain_graph=(
                position < len(selected_idx) - 1
            ),
            create_graph=False,
            allow_unused=True
        )

        grad_vector = flatten_gradients(
            grads,
            prompt_params
        )

        grad_vectors.append(grad_vector)

    if len(grad_vectors) < 2:
        return {
            "gradient_agreement": float("nan"),
            "negative_gradient_ratio": float("nan"),
            "consensus_agreement": float("nan"),
        }

    gradients = torch.stack(grad_vectors)

    grad_norms = gradients.norm(
        dim=1,
        keepdim=True
    ).clamp_min(1e-12)

    normalized_gradients = gradients / grad_norms

    # 两两 cosine similarity
    cosine_matrix = (
        normalized_gradients
        @ normalized_gradients.t()
    )

    mask = ~torch.eye(
        cosine_matrix.size(0),
        dtype=torch.bool,
        device=cosine_matrix.device
    )

    pairwise_cosines = cosine_matrix[mask]

    gradient_agreement = pairwise_cosines.mean()

    negative_gradient_ratio = (
        pairwise_cosines < 0
    ).float().mean()

    # 每个视图与平均更新方向的相似度
    consensus_gradient = gradients.mean(dim=0)

    consensus_gradient = (
        consensus_gradient
        / consensus_gradient.norm().clamp_min(1e-12)
    )

    consensus_cosines = (
        normalized_gradients
        @ consensus_gradient
    )

    return {
        "gradient_agreement":
            gradient_agreement.item(),

        "negative_gradient_ratio":
            negative_gradient_ratio.item(),

        "consensus_agreement":
            consensus_cosines.mean().item(),
    }
def test_time_tuning(model, inputs, optimizer, scaler, args):
    """
    Pure/original R-TPT prompt tuning.

    No Dirichlet.
    No SC view filtering.
    No extra consistency loss.

    If args.tta_steps == 0, this function is simply skipped by the caller.
    """
    selected_idx = None

    for _ in range(args.tta_steps):
        output = model(inputs)

        if selected_idx is not None:
            output = output[selected_idx]
        else:
            output, selected_idx = select_confident_samples(
                output,
                args.selection_p
            )

        loss = entropy_avg(output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def collect_agreement_statistics(
    model,
    images,
    args
):
    # Prediction agreement
    with torch.no_grad():
        logits = model(images)

        all_prediction_stats = (
            prediction_agreement(logits)
        )

        selected_logits, selected_idx = (
            select_confident_samples(
                logits,
                args.selection_p
            )
        )

        selected_prediction_stats = (
            prediction_agreement(
                selected_logits
            )
        )

    # Prompt-gradient agreement
    gradient_stats = (
        prompt_gradient_agreement(
            model=model,
            inputs=images,
            selection_p=args.selection_p,
            max_views=args.diag_grad_views
        )
    )

    return {
        "pred_agreement_all":
            all_prediction_stats[
                "top1_agreement"
            ],

        "prob_agreement_all":
            all_prediction_stats[
                "prob_agreement"
            ],

        "unique_predictions_all":
            all_prediction_stats[
                "num_unique_predictions"
            ],

        "pred_agreement_selected":
            selected_prediction_stats[
                "top1_agreement"
            ],

        "prob_agreement_selected":
            selected_prediction_stats[
                "prob_agreement"
            ],

        "unique_predictions_selected":
            selected_prediction_stats[
                "num_unique_predictions"
            ],

        **gradient_stats
    }
def compute_ece(confidences, corrects, n_bins=15):
    """
    Expected Calibration Error (ECE), returned in percentage points.

    confidences: Tensor [N], max softmax confidence in [0, 1]
    corrects:    Tensor [N], bool or 0/1
    """
    confidences = confidences.float()
    corrects = corrects.float()

    bin_boundaries = torch.linspace(
        0,
        1,
        n_bins + 1,
        device=confidences.device
    )

    ece = torch.zeros(1, device=confidences.device)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            in_bin = (
                (confidences > lower)
                & (confidences <= upper)
            )

        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            acc_in_bin = corrects[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()

            ece += (
                torch.abs(conf_in_bin - acc_in_bin)
                * prop_in_bin
            )

    return ece.item() * 100.0


def safe_mean(x):
    if x.numel() == 0:
        return float("nan")
    return x.float().mean().item()


def summarize_calibration_stats(
    confidences,
    corrects,
    high_conf_th=0.9,
    n_bins=15,
):
    """
    Compute all requested calibration metrics.

    Returns:
        Acc
        ECE
        AvgConf
        WrongConf
        CorrectConf
        High-confidence wrong rate over all samples
        High-confidence rate among wrong samples
    """
    confidences = confidences.float()
    corrects = corrects.bool()

    wrongs = ~corrects

    acc = corrects.float().mean().item() * 100.0
    ece = compute_ece(
        confidences,
        corrects,
        n_bins=n_bins
    )

    avg_conf = confidences.mean().item() * 100.0
    wrong_conf = safe_mean(confidences[wrongs]) * 100.0
    correct_conf = safe_mean(confidences[corrects]) * 100.0

    wrong_high_conf_rate_all = (
        (
            (confidences >= high_conf_th)
            & wrongs
        )
        .float()
        .mean()
        .item()
        * 100.0
    )

    if wrongs.sum().item() > 0:
        wrong_high_conf_rate_wrong_only = (
            (
                confidences[wrongs]
                >= high_conf_th
            )
            .float()
            .mean()
            .item()
            * 100.0
        )
    else:
        wrong_high_conf_rate_wrong_only = float("nan")

    return {
        "acc": acc,
        "ece": ece,
        "avg_conf": avg_conf,
        "wrong_conf": wrong_conf,
        "correct_conf": correct_conf,
        "wrong_high_conf_rate_all": wrong_high_conf_rate_all,
        "wrong_high_conf_rate_wrong_only": (
            wrong_high_conf_rate_wrong_only
        ),
    }


def format_stats(stats, name):
    lines = [
        "========== {} ==========".format(name),
        "Acc: {:.4f}".format(stats["acc"]),
        "ECE: {:.4f}".format(stats["ece"]),
        "AvgConf: {:.4f}".format(stats["avg_conf"]),
        "WrongConf: {:.4f}".format(stats["wrong_conf"]),
        "CorrectConf: {:.4f}".format(stats["correct_conf"]),
        (
            "High-confidence wrong (all samples): "
            "{:.4f}%"
        ).format(stats["wrong_high_conf_rate_all"]),
        (
            "High-confidence among wrong samples: "
            "{:.4f}%"
        ).format(
            stats["wrong_high_conf_rate_wrong_only"]
        ),
    ]

    return "\n".join(lines)


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    args.alpha = args.eps / 4.0

    args.output_dir = os.path.join(
        args.output_dir,
        args.arch,
        args.test_sets,
        "eps_{}_alpha_{}_step_{}_tta_{}".format(
            args.eps,
            args.alpha,
            args.steps,
            args.tta_steps,
        ),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    args.out_file = open(
        os.path.join(args.output_dir, "log.txt"),
        "w",
    )

    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()

    assert args.gpu is not None

    set_random_seed(args.seed)
    print("Use GPU: {} for evaluation".format(args.gpu))

    # ============================================================
    # Model and class names
    # ============================================================
    dset = args.test_sets

    if len(dset) > 1:
        classnames = eval(
            "{}_classes".format(dset.lower())
        )
    else:
        assert dset in ["A", "R", "K", "V", "I"]

        classnames_all = imagenet_classes
        classnames = []

        if dset in ["A", "R", "V"]:
            label_mask = eval(
                "imagenet_{}_mask".format(
                    dset.lower()
                )
            )

            if dset == "R":
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(
                            classnames_all[i]
                        )
            else:
                classnames = [
                    classnames_all[i]
                    for i in label_mask
                ]
        else:
            classnames = classnames_all

    args.classnames = classnames

    model = get_coop(
        args.arch,
        classnames,
        args.gpu,
        args.n_ctx,
        args.ctx_init,
    )

    model_state = None

    # ============================================================
    # Optional robust encoder loading
    # ============================================================
    if len(args.load_tecoa) > 0:
        args.robust_pretrain_path = {
            "RN50-eps1": (
                "pretrain/tecoa/"
                "rn50_eps1.pth.tar"
            ),
        }[args.load_tecoa]

        robust_state_dict = torch.load(
            args.robust_pretrain_path,
            map_location="cpu",
        )

        model.image_encoder.load_state_dict(
            robust_state_dict[
                "vision_encoder_state_dict"
            ]
        )

        print("load robust vision encoder")

    # Only prompt parameters are trainable.
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print(
        "=> Model created: visual backbone {}".format(
            args.arch
        )
    )

    if not torch.cuda.is_available():
        print("using CPU, this will be slow")
    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    trainable_param = model.prompt_learner.parameters()

    optimizer = torch.optim.AdamW(
        trainable_param,
        args.lr,
    )

    optim_state = deepcopy(
        optimizer.state_dict()
    )

    scaler = None
    cudnn.benchmark = True

    # Kept from the original code for compatibility.
    normalize = transforms.Normalize(
        mean=[
            0.48145466,
            0.4578275,
            0.40821073,
        ],
        std=[
            0.26862954,
            0.26130258,
            0.27577711,
        ],
    )

    # ============================================================
    # Dataset
    # ============================================================
    base_transform = transforms.Compose(
        [
            transforms.Resize(
                args.resolution,
                interpolation=BICUBIC,
            ),
            transforms.CenterCrop(
                args.resolution
            ),
        ]
    )

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    data_transform = AugMixAugmenter(
        base_transform,
        preprocess,
        n_views=args.batch_size - 1,
        augmix=len(dset) > 1,
    )

    batchsize = 1

    val_dataset = build_dataset(
        dset,
        data_transform,
        args.data,
        mode=args.dataset_mode,
    )

    print(
        "number of test samples: {}".format(
            len(val_dataset)
        )
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("evaluating: {}".format(dset))

    results = test_time_adapt_eval(
        val_loader,
        model,
        model_state,
        optimizer,
        optim_state,
        scaler,
        args,
        data_transform,
    )

    del val_dataset, val_loader

    clip_stats = results["clip_stats"]

    prompt_only_stats = results[
        "prompt_only_stats"
    ]

    rtpt_stats = results["rtpt_stats"]

    mode_name = (
        "Clean"
        if args.eps <= 0
        else "Robust"
    )

    clip_log = format_stats(
        clip_stats,
        "CLIP {} / TTA Step 0 Baseline".format(
            mode_name
        ),
    )

    prompt_only_log = format_stats(
        prompt_only_stats,
        (
            "Prompt Tuning + Uniform Ensemble "
            "{} / TTA Step {}"
        ).format(
            mode_name,
            args.tta_steps,
        ),
    )

    rtpt_log = format_stats(
        rtpt_stats,
        "R-TPT {} / TTA Step {}".format(
            mode_name,
            args.tta_steps,
        ),
    )

    comparison_log = (
        "========== Comparison ==========\n"
        "Prompt-Only vs CLIP Acc: {:+.4f}\n"
        "R-TPT vs CLIP Acc: {:+.4f}\n"
        "R-TPT vs Prompt-Only Acc: {:+.4f}\n"
        "R-TPT vs CLIP ECE: {:+.4f}\n"
        "R-TPT vs CLIP AvgConf: {:+.4f}\n"
        "R-TPT vs CLIP WrongConf: {:+.4f}\n"
        "R-TPT vs CLIP CorrectConf: {:+.4f}\n"
        "R-TPT vs CLIP high-confidence wrong: {:+.4f}"
    ).format(
        prompt_only_stats["acc"]
        - clip_stats["acc"],

        rtpt_stats["acc"]
        - clip_stats["acc"],

        rtpt_stats["acc"]
        - prompt_only_stats["acc"],

        rtpt_stats["ece"]
        - clip_stats["ece"],

        rtpt_stats["avg_conf"]
        - clip_stats["avg_conf"],

        rtpt_stats["wrong_conf"]
        - clip_stats["wrong_conf"],

        rtpt_stats["correct_conf"]
        - clip_stats["correct_conf"],

        rtpt_stats["wrong_high_conf_rate_all"]
        - clip_stats["wrong_high_conf_rate_all"],
    )

    final_log = (
        clip_log
        + "\n\n"
        + prompt_only_log
        + "\n\n"
        + rtpt_log
        + "\n\n"
        + comparison_log
    )

    args.out_file.write(
        final_log + "\n"
    )

    args.out_file.flush()
    print(final_log + "\n")

    save_log = {
    "mode": mode_name,
    "tta_steps": args.tta_steps,

    "clip_stats": clip_stats,
    "prompt_only_stats": prompt_only_stats,
    "rtpt_stats": rtpt_stats,

    "comparison": {
        "prompt_only_vs_clip_acc": (
            prompt_only_stats["acc"]
            - clip_stats["acc"]
        ),
        "rtpt_vs_clip_acc": (
            rtpt_stats["acc"]
            - clip_stats["acc"]
        ),
        "rtpt_vs_prompt_only_acc": (
            rtpt_stats["acc"]
            - prompt_only_stats["acc"]
        ),
    },
}

    torch.save(
        save_log,
        os.path.join(
            args.output_dir,
            "results_log.pt",
        ),
    )

    args.out_file.close()


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
    batch_time = AverageMeter(
        "Time",
        ":6.3f",
        Summary.NONE,
    )

    top1 = AverageMeter(
        "Acc@1",
        ":6.2f",
        Summary.AVERAGE,
    )

    tpt1 = AverageMeter(
        "TTAAcc@1",
        ":6.2f",
        Summary.AVERAGE,
    )
    prompt_only1 = AverageMeter(
    "PromptOnlyAcc@1",
    ":6.2f",
    Summary.AVERAGE,
    )
    progress = ProgressMeter(
        len(val_loader),
        [
            batch_time,
            top1,
            prompt_only1,
            tpt1,
        ],
        prefix="Test: ",
    )

    # ============================================================
    # Calibration records
    # ============================================================
    clip_conf_list = []
    clip_correct_list = []
    clip_pred_list = []

    # 只进行 prompt tuning，不进行 reliability weighting
    prompt_only_conf_list = []
    prompt_only_correct_list = []
    prompt_only_pred_list = []

    rtpt_conf_list = []
    rtpt_correct_list = []
    rtpt_pred_list = []

    label_list = []

    # ============================================================
    # Evaluation setup
    # ============================================================
    model.eval()
    diagnostic_results = []

    if args.eps > 0.0:
        assert args.steps > 0

        atk = torchattacks.PGD(
            model,
            eps=args.eps / 255,
            alpha=args.alpha / 255,
            steps=args.steps,
        )

    end = time.time()

    for i, (images, target) in enumerate(
        val_loader
    ):
        assert args.gpu is not None

        target = target.cuda(
            args.gpu,
            non_blocking=True,
        )

        # ========================================================
        # Optional PGD adversarial branch
        # ========================================================
        if args.eps > 0.0:
            image = images[0].cuda(
                args.gpu,
                non_blocking=True,
            )

            adv_image = atk(
                image,
                target,
            )

            img_adv = transforms.ToPILImage()(
                adv_image
                .squeeze(0)
                .detach()
                .cpu()
            )

            images = data_transform(
                img_adv
            )

            images = [
                _.unsqueeze(0)
                for _ in images
            ]

        # ========================================================
        # Move views to GPU
        # ========================================================
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(
                    args.gpu,
                    non_blocking=True,
                )

            image = images[0]

        else:
            if len(images.size()) > 4:
                assert (
                    images.size()[0] == 1
                )
                images = images.squeeze(0)

            images = images.cuda(
                args.gpu,
                non_blocking=True,
            )

            image = images

        images = torch.cat(
            images,
            dim=0,
        )

        # ========================================================
        # Reset prompt and optimizer for every test sample
        # ========================================================
        with torch.no_grad():
            model.reset()

        optimizer.load_state_dict(
            optim_state
        )

        # ========================================================
        # TTA Step 0 baseline:
        # original single-view CLIP prediction
        #
        # Also extract pre-TTA features used by original R-TPT
        # reliability-weighted ensemble.
        # ========================================================
        with torch.no_grad():
            clip_output = model(image)

            clip_features, _, _ = (
                model.forward_features(images)
            )

            untuned_outputs = model(images)


# ========================================================
# Agreement diagnostics：必须放在 prompt tuning 之前
# ========================================================
        diagnostics = None

        if (
            args.diagnostics
            and i < args.diag_max_samples
        ):
            diagnostics = collect_agreement_statistics(
                model=model,
                images=images,
                args=args,
            )

            # 防御性清理；autograd.grad 通常不会写入 param.grad
            optimizer.zero_grad(set_to_none=True)


        if args.tta_steps > 0:
            test_time_tuning(
                model,
                images,
                optimizer,
                scaler,
                args,
            )

            with torch.no_grad():
                tuned_outputs = model(images)
        else:
            tuned_outputs = untuned_outputs


        # ========================================================
        # Prompt-Only
        #
        # 使用所有视图，但不使用 reliability weighting。
        # 每个视图具有相同权重。
        # ========================================================
        prompt_only_output = tuned_outputs.mean(
            dim=0,
            keepdim=True,
        )
                

        # ========================================================
        # Original R-TPT reliability-weighted ensemble
        # ========================================================
        sim_matrix_images = torch.bmm(
            clip_features.unsqueeze(0),
            clip_features
            .unsqueeze(0)
            .permute(0, 2, 1),
        )

        score = get_top_sim(
            sim_matrix_images
        )

        weight = torch.nn.functional.softmax(
            score / args.rtpt_tau,
            dim=-1,
        )

        rtpt_output = torch.sum(
            weight.unsqueeze(-1) * tuned_outputs,
            dim=0,
            keepdim=True,
        )

        # ========================================================
        # Accuracy
        # ========================================================
        acc1, _ = accuracy(
            clip_output,
            target,
            topk=(1, 5),
        )

        prompt_only_acc1, _ = accuracy(
            prompt_only_output,
            target,
            topk=(1, 5),
        )

        rtpt_acc1, _ = accuracy(
            rtpt_output,
            target,
            topk=(1, 5),
        )

        top1.update(
            acc1[0],
            1,
        )

        prompt_only1.update(
            prompt_only_acc1[0],
            1,
        )

        tpt1.update(
            rtpt_acc1[0],
            1,
        )
            # ========================================================
        # Per-sample calibration records
        # ========================================================
        with torch.no_grad():
            # =========================
            # CLIP
            # =========================
            clip_prob = torch.softmax(
                clip_output,
                dim=1,
            )

            clip_conf, clip_pred = (
                clip_prob.max(dim=1)
            )

            clip_correct = clip_pred.eq(target)

            # =========================
            # Prompt tuning + uniform ensemble
            # =========================
            prompt_only_prob = torch.softmax(
                prompt_only_output,
                dim=1,
            )

            prompt_only_conf, prompt_only_pred = (
                prompt_only_prob.max(dim=1)
            )

            prompt_only_correct = (
                prompt_only_pred.eq(target)
            )

            # =========================
            # Full R-TPT
            # =========================
            rtpt_prob = torch.softmax(
                rtpt_output,
                dim=1,
            )

            rtpt_conf, rtpt_pred = (
                rtpt_prob.max(dim=1)
            )

            rtpt_correct = rtpt_pred.eq(target)

            # =========================
            # Save per-sample outputs
            # =========================
            clip_conf_list.append(
                clip_conf.detach().cpu()
            )
            clip_correct_list.append(
                clip_correct.detach().cpu()
            )
            clip_pred_list.append(
                clip_pred.detach().cpu()
            )

            prompt_only_conf_list.append(
                prompt_only_conf.detach().cpu()
            )
            prompt_only_correct_list.append(
                prompt_only_correct.detach().cpu()
            )
            prompt_only_pred_list.append(
                prompt_only_pred.detach().cpu()
            )

            rtpt_conf_list.append(
                rtpt_conf.detach().cpu()
            )
            rtpt_correct_list.append(
                rtpt_correct.detach().cpu()
            )
            rtpt_pred_list.append(
                rtpt_pred.detach().cpu()
            )

            label_list.append(
                target.detach().cpu()
            )

            if diagnostics is not None:
                diagnostic_results.append({
                    "sample_index": int(i),
                    "eps": float(args.eps),
                    "target": int(target.item()),

                    "clip_prediction": int(
                        clip_pred.item()
                    ),
                    "clip_correct": int(
                        clip_correct.item()
                    ),

                    "prompt_only_prediction": int(
                        prompt_only_pred.item()
                    ),
                    "prompt_only_correct": int(
                        prompt_only_correct.item()
                    ),

                    "rtpt_prediction": int(
                        rtpt_pred.item()
                    ),
                    "rtpt_correct": int(
                        rtpt_correct.item()
                    ),

                    **diagnostics,
                })

        # ========================================================
        # Timing and logging
        # ========================================================
        batch_time.update(
            time.time() - end
        )

        end = time.time()

        if (
            (i + 1) % args.print_freq == 0
            or (i + 1) == len(val_loader)
        ):
            if args.eps <= 0:
                print_log = (
                    "iter:{}/{}, "
                    "clip_acc1={}, "
                    "rtpt_tta_step_{}_acc1={}"
                ).format(
                    i + 1,
                    len(val_loader),
                    top1.avg,
                    args.tta_steps,
                    tpt1.avg,
                )
            else:
                print_log = (
                    "iter:{}/{}, "
                    "clip_adv1={}, "
                    "rtpt_tta_step_{}_adv1={}"
                ).format(
                    i + 1,
                    len(val_loader),
                    top1.avg,
                    args.tta_steps,
                    tpt1.avg,
                )

            args.out_file.write(
                print_log + "\n"
            )

            args.out_file.flush()
            print(print_log + "\n")

            progress.display(i)

    progress.display_summary()

    # ============================================================
    # Aggregate calibration analysis
    # ============================================================
    clip_conf_all = torch.cat(
        clip_conf_list
    )

    clip_correct_all = torch.cat(
        clip_correct_list
    )

    clip_pred_all = torch.cat(
        clip_pred_list
    )
    prompt_only_conf_all = torch.cat(
        prompt_only_conf_list
    )

    prompt_only_correct_all = torch.cat(
        prompt_only_correct_list
    )

    prompt_only_pred_all = torch.cat(
        prompt_only_pred_list
    )

    rtpt_conf_all = torch.cat(
        rtpt_conf_list
    )

    rtpt_correct_all = torch.cat(
        rtpt_correct_list
    )

    rtpt_pred_all = torch.cat(
        rtpt_pred_list
    )

    label_all = torch.cat(
        label_list
    )

    clip_stats = summarize_calibration_stats(
        confidences=clip_conf_all,
        corrects=clip_correct_all,
        high_conf_th=args.high_conf_th,
        n_bins=args.ece_bins,
    )
    prompt_only_stats = summarize_calibration_stats(
    confidences=prompt_only_conf_all,
    corrects=prompt_only_correct_all,
    high_conf_th=args.high_conf_th,
    n_bins=args.ece_bins,
    )

    rtpt_stats = summarize_calibration_stats(
        confidences=rtpt_conf_all,
        corrects=rtpt_correct_all,
        high_conf_th=args.high_conf_th,
        n_bins=args.ece_bins,
    )

    # ============================================================
    # Save per-sample values for later plots/analysis
    # ============================================================
    calibration_save = {
        "tta_steps": args.tta_steps,
        "clip_conf": clip_conf_all.numpy(),
        "clip_correct": (
            clip_correct_all.numpy()
        ),
        "clip_pred": clip_pred_all.numpy(),
        "prompt_only_conf": (
            prompt_only_conf_all.numpy()
        ),

        "prompt_only_correct": (
            prompt_only_correct_all.numpy()
        ),

        "prompt_only_pred": (
            prompt_only_pred_all.numpy()
        ),

        
        "rtpt_conf": rtpt_conf_all.numpy(),
        "rtpt_correct": (
            rtpt_correct_all.numpy()
        ),
        "rtpt_pred": rtpt_pred_all.numpy(),
        "label": label_all.numpy(),
        "clip_stats": clip_stats,
        "prompt_only_stats": prompt_only_stats,
        "rtpt_stats": rtpt_stats,
    }

    torch.save(
        calibration_save,
        os.path.join(
            args.output_dir,
            "calibration_analysis.pt",
        ),
    )
    if args.diagnostics:
        diagnostic_path = os.path.join(
            args.output_dir,
            "agreement_diagnostics.pt",
        )

        torch.save(
            diagnostic_results,
            diagnostic_path,
        )

        print(
            "Agreement diagnostics saved to:",
            diagnostic_path,
        )
    return {
        "clip_stats": clip_stats,
        "prompt_only_stats": prompt_only_stats,
        "rtpt_stats": rtpt_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Pure R-TPT with calibration analysis"
        )
    )

    parser.add_argument(
        "data",
        metavar="DIR",
        help="path to dataset root",
    )

    parser.add_argument(
        "--test_sets",
        type=str,
        default="Caltech101",
    )

    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="test",
    )

    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="RN50",
    )

    parser.add_argument(
        "--resolution",
        default=224,
        type=int,
        help="CLIP image resolution",
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help=(
            "number of data loading workers "
            "(default: 4)"
        ),
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        default=64,
        type=int,
        metavar="N",
    )

    parser.add_argument(
        "-p",
        "--print-freq",
        default=200,
        type=int,
        metavar="N",
        help="print frequency",
    )

    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="GPU id to use.",
    )

    parser.add_argument(
        "--n_ctx",
        default=4,
        type=int,
        help="number of tunable tokens",
    )

    parser.add_argument(
        "--ctx_init",
        default=None,
        type=str,
        help="init tunable prompts",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "output_results/"
            "rtpt_calibration"
        ),
    )

    parser.add_argument(
        "--eps",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--alpha",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=5e-3,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )

    parser.add_argument(
        "--selection_p",
        default=0.1,
        type=float,
        help=(
            "fraction of low-entropy views "
            "used for prompt tuning"
        ),
    )

    parser.add_argument(
        "--tta_steps",
        default=1,
        type=int,
        help=(
            "number of test-time adaptation steps; "
            "0 means no prompt update"
        ),
    )

    parser.add_argument(
        "--load_tecoa",
        type=str,
        default="",
        choices=[
            "",
            "RN50-eps1",
            "ViT-B/32-eps1",
            "ViT-B/32-eps4",
        ],
    )

    parser.add_argument(
        "--ece_bins",
        type=int,
        default=15,
        help="number of bins for ECE",
    )

    parser.add_argument(
        "--high_conf_th",
        type=float,
        default=0.9,
        help=(
            "threshold for high-confidence "
            "wrong predictions"
        ),
    )

    parser.add_argument(
        "--rtpt_tau",
        type=float,
        default=0.01,
        help=(
            "temperature for original R-TPT "
            "view-ensemble weights"
        ),
    )

    parser.add_argument(
    "--diagnostics",
    action="store_true",
    help="collect view agreement diagnostics",
    )

    parser.add_argument(
        "--diag_max_samples",
        type=int,
        default=200,
        help="maximum number of diagnostic samples",
    )

    parser.add_argument(
        "--diag_grad_views",
        type=int,
        default=6,
        help="number of selected views used for gradient agreement",
    )

    main()