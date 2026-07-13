#!/bin/bash
#SBATCH --job-name=download_datasets
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=download_datasets_%j.out
#SBATCH --error=download_datasets_%j.err

set -Eeuo pipefail

module add anaconda3/2022.05
source activate rtpt

DATA="/scratch/hpc/07/zhang303/R-TPT-main/dataset"
TMP_DIR="${DATA}/.downloads"

mkdir -p "$DATA" "$TMP_DIR"

log() {
    echo
    echo "[$(date '+%F %T')] $*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

download_http() {
    local url="$1"
    local output="$2"

    if [[ -s "$output" ]]; then
        log "文件已经存在，跳过下载：$output"
        return
    fi

    if command -v curl >/dev/null 2>&1; then
        curl -fL \
            --retry 5 \
            --retry-delay 5 \
            --continue-at - \
            "$url" \
            -o "$output"
    elif command -v wget >/dev/null 2>&1; then
        wget \
            --continue \
            --tries=5 \
            --waitretry=5 \
            "$url" \
            -O "$output"
    else
        die "系统中没有 curl 或 wget"
    fi
}

ensure_gdown() {
    if command -v gdown >/dev/null 2>&1; then
        GDOWN=(gdown)
    elif python -c "import gdown" >/dev/null 2>&1; then
        GDOWN=(python -m gdown)
    else
        log "当前 rtpt 环境中没有 gdown，开始安装"
        python -m pip install --upgrade gdown
        GDOWN=(python -m gdown)
    fi
}

download_gdrive() {
    local file_id="$1"
    local output="$2"

    if [[ -s "$output" ]]; then
        log "文件已经存在，跳过下载：$output"
        return
    fi

    ensure_gdown

    "${GDOWN[@]}" \
        --fuzzy \
        "https://drive.google.com/file/d/${file_id}/view" \
        -O "$output"
}

verify_file() {
    local file="$1"

    if [[ ! -s "$file" ]]; then
        die "文件不存在或为空：$file"
    fi
}

log "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
log "运行节点：$(hostname)"
log "Python 路径：$(which python)"
log "Conda 环境：${CONDA_DEFAULT_ENV:-unknown}"
log "数据集目录：$DATA"



# ============================================================
# Caltech101
# ============================================================

CALTECH_DIR="${DATA}/caltech-101"
CALTECH_ZIP="${TMP_DIR}/caltech-101.zip"
CALTECH_TMP="${TMP_DIR}/caltech101_extract"

mkdir -p "$CALTECH_DIR"

log "开始处理 Caltech101"

if [[ ! -d "${CALTECH_DIR}/101_ObjectCategories" ]]; then

    download_http \
        "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1" \
        "$CALTECH_ZIP"

    verify_file "$CALTECH_ZIP"

    log "开始解压 Caltech101 外层 ZIP"

    rm -rf "$CALTECH_TMP"
    mkdir -p "$CALTECH_TMP"

    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$CALTECH_ZIP" -d "$CALTECH_TMP"
    else
        python -m zipfile -e "$CALTECH_ZIP" "$CALTECH_TMP"
    fi

    INNER_TAR="$(
        find "$CALTECH_TMP" \
            -type f \
            -name "101_ObjectCategories.tar.gz" \
            ! -path "*/__MACOSX/*" \
            -print \
            -quit
    )"

    if [[ -z "$INNER_TAR" ]]; then
        echo "外层 ZIP 解压后的内容："
        find "$CALTECH_TMP" -maxdepth 4 -print
        die "没有找到内层 101_ObjectCategories.tar.gz"
    fi

    log "开始解压内层 101_ObjectCategories.tar.gz"

    tar -xzf "$INNER_TAR" -C "$CALTECH_DIR"

    if [[ ! -d "${CALTECH_DIR}/101_ObjectCategories" ]]; then
        die "内层压缩包解压后仍未找到 101_ObjectCategories"
    fi

    rm -rf "$CALTECH_TMP"
else
    log "Caltech101 已经解压，跳过"
fi

download_gdrive \
    "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN" \
    "${CALTECH_DIR}/split_zhou_Caltech101.json"

# ============================================================
# DTD
# ============================================================

DTD_DIR="${DATA}/dtd"
DTD_ARCHIVE="${TMP_DIR}/dtd-r1.0.1.tar.gz"

log "开始处理 DTD"

if [[ ! -d "${DTD_DIR}/images" ]]; then
    download_http \
        "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" \
        "$DTD_ARCHIVE"

    verify_file "$DTD_ARCHIVE"

    log "开始解压 DTD"
    tar -xzf "$DTD_ARCHIVE" -C "$DATA"
else
    log "DTD 已经解压，跳过"
fi

mkdir -p "$DTD_DIR"

download_gdrive \
    "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x" \
    "${DTD_DIR}/split_zhou_DescribableTextures.json"

# ============================================================
# UCF101
# ============================================================

UCF_DIR="${DATA}/ucf101"
UCF_ARCHIVE="${TMP_DIR}/UCF-101-midframes.zip"

mkdir -p "$UCF_DIR"

log "开始处理 UCF101"

if [[ ! -d "${UCF_DIR}/UCF-101-midframes" ]]; then
    download_gdrive \
        "10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O" \
        "$UCF_ARCHIVE"

    verify_file "$UCF_ARCHIVE"

    log "开始解压 UCF101"

    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$UCF_ARCHIVE" -d "$UCF_DIR"
    else
        python -m zipfile -e "$UCF_ARCHIVE" "$UCF_DIR"
    fi
else
    log "UCF101 已经解压，跳过"
fi

download_gdrive \
    "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y" \
    "${UCF_DIR}/split_zhou_UCF101.json"

# ============================================================
# 检查目录和文件
# ============================================================

log "开始检查数据集目录结构"

[[ -d "${CALTECH_DIR}/101_ObjectCategories" ]] \
    || die "缺少 ${CALTECH_DIR}/101_ObjectCategories"

verify_file "${CALTECH_DIR}/split_zhou_Caltech101.json"

[[ -d "${DTD_DIR}/images" ]] \
    || die "缺少 ${DTD_DIR}/images"

[[ -d "${DTD_DIR}/imdb" ]] \
    || die "缺少 ${DTD_DIR}/imdb"

[[ -d "${DTD_DIR}/labels" ]] \
    || die "缺少 ${DTD_DIR}/labels"

verify_file "${DTD_DIR}/split_zhou_DescribableTextures.json"

[[ -d "${UCF_DIR}/UCF-101-midframes" ]] \
    || die "缺少 ${UCF_DIR}/UCF-101-midframes"

verify_file "${UCF_DIR}/split_zhou_UCF101.json"

log "三个数据集均已下载并解压完成"

echo
echo "最终目录结构："
find "$DATA" \
    -maxdepth 2 \
    -mindepth 1 \
    \( -path "$TMP_DIR" -o -path "$TMP_DIR/*" \) -prune -o \
    -print | sort