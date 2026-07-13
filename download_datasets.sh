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
CALTECH_ARCHIVE="${TMP_DIR}/caltech-101.zip"
CALTECH_TMP="${TMP_DIR}/caltech101_extract"

mkdir -p "$CALTECH_DIR"

log "开始处理 Caltech101"

if [[ ! -d "${CALTECH_DIR}/101_ObjectCategories" ]]; then

    download_http \
        "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1" \
        "$CALTECH_ARCHIVE"

    verify_file "$CALTECH_ARCHIVE"

    log "开始解压 Caltech101"

    rm -rf "$CALTECH_TMP"
    mkdir -p "$CALTECH_TMP"

    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$CALTECH_ARCHIVE" -d "$CALTECH_TMP"
    else
        python -m zipfile -e "$CALTECH_ARCHIVE" "$CALTECH_TMP"
    fi

    FOUND_CALTECH_DIR="$(
        find "$CALTECH_TMP" \
            -type d \
            -name "101_ObjectCategories" \
            -print \
            -quit
    )"

    if [[ -z "$FOUND_CALTECH_DIR" ]]; then
        echo "解压后的目录结构："
        find "$CALTECH_TMP" -maxdepth 4 -print
        die "解压后没有找到 101_ObjectCategories"
    fi

    rm -rf "${CALTECH_DIR}/101_ObjectCategories"

    mv "$FOUND_CALTECH_DIR" \
        "${CALTECH_DIR}/101_ObjectCategories"

    rm -rf "$CALTECH_TMP"
else
    log "Caltech101 已经解压，跳过"
fi

download_gdrive \
    "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN" \
    "${CALTECH_DIR}/split_zhou_Caltech101.json"