#!/bin/bash
#SBATCH --job-name=download_data2
#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=download_data2_%j.out
#SBATCH --error=download_data2_%j.err

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

verify_file() {
    local file="$1"

    if [[ ! -s "$file" ]]; then
        die "文件不存在或为空：$file"
    fi
}

download_http() {
    local url="$1"
    local output="$2"

    if [[ -s "$output" ]]; then
        log "文件已经存在，跳过下载：$output"
        return
    fi

    mkdir -p "$(dirname "$output")"

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

    mkdir -p "$(dirname "$output")"

    ensure_gdown

    "${GDOWN[@]}" \
        --fuzzy \
        "https://drive.google.com/file/d/${file_id}/view" \
        -O "$output"
}

log "SLURM Job ID：${SLURM_JOB_ID:-N/A}"
log "运行节点：$(hostname)"
log "Python 路径：$(which python)"
log "Conda 环境：${CONDA_DEFAULT_ENV:-unknown}"
log "数据集目录：$DATA"

# ============================================================
# OxfordPets
# ============================================================

PETS_DIR="${DATA}/oxford_pets"
PETS_IMAGES_ARCHIVE="${TMP_DIR}/oxford_pets_images.tar.gz"
PETS_ANNOTATIONS_ARCHIVE="${TMP_DIR}/oxford_pets_annotations.tar.gz"

mkdir -p "$PETS_DIR"

log "开始处理 OxfordPets"

if [[ ! -d "${PETS_DIR}/images" ]]; then
    download_http \
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz" \
        "$PETS_IMAGES_ARCHIVE"

    verify_file "$PETS_IMAGES_ARCHIVE"

    log "开始解压 OxfordPets images"
    tar -xzf "$PETS_IMAGES_ARCHIVE" -C "$PETS_DIR"
else
    log "OxfordPets images 已经解压，跳过"
fi

if [[ ! -d "${PETS_DIR}/annotations" ]]; then
    download_http \
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz" \
        "$PETS_ANNOTATIONS_ARCHIVE"

    verify_file "$PETS_ANNOTATIONS_ARCHIVE"

    log "开始解压 OxfordPets annotations"
    tar -xzf "$PETS_ANNOTATIONS_ARCHIVE" -C "$PETS_DIR"
else
    log "OxfordPets annotations 已经解压，跳过"
fi

download_gdrive \
    "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs" \
    "${PETS_DIR}/split_zhou_OxfordPets.json"

# ============================================================
# Flowers102
# ============================================================

FLOWERS_DIR="${DATA}/oxford_flowers"
FLOWERS_IMAGES_ARCHIVE="${TMP_DIR}/102flowers.tgz"

mkdir -p "$FLOWERS_DIR"

log "开始处理 Flowers102"

if [[ ! -d "${FLOWERS_DIR}/jpg" ]]; then
    download_http \
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz" \
        "$FLOWERS_IMAGES_ARCHIVE"

    verify_file "$FLOWERS_IMAGES_ARCHIVE"

    log "开始解压 Flowers102 images"
    tar -xzf "$FLOWERS_IMAGES_ARCHIVE" -C "$FLOWERS_DIR"
else
    log "Flowers102 images 已经解压，跳过"
fi

download_http \
    "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat" \
    "${FLOWERS_DIR}/imagelabels.mat"

download_gdrive \
    "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0" \
    "${FLOWERS_DIR}/cat_to_name.json"

download_gdrive \
    "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT" \
    "${FLOWERS_DIR}/split_zhou_OxfordFlowers.json"

# ============================================================
# Food101
# ============================================================
# ============================================================
# Food101
# ============================================================

FOOD_ARCHIVE="${TMP_DIR}/food-101.tar.gz"
FOOD_DIR="${DATA}/food-101"

log "开始处理 Food101"

if [[ ! -d "${FOOD_DIR}/images" ]]; then
    download_http \
        "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz" \
        "$FOOD_ARCHIVE"

    verify_file "$FOOD_ARCHIVE"

    log "开始解压 Food101"
    tar -xzf "$FOOD_ARCHIVE" -C "$DATA"
else
    log "Food101 已经解压，跳过"
fi

download_gdrive \
    "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl" \
    "${FOOD_DIR}/split_zhou_Food101.json"

# ============================================================
# FGVCAircraft
# ============================================================

AIRCRAFT_ARCHIVE="${TMP_DIR}/fgvc-aircraft-2013b.tar.gz"
AIRCRAFT_TMP="${TMP_DIR}/fgvc_aircraft_extract"
AIRCRAFT_DIR="${DATA}/fgvc_aircraft"

log "开始处理 FGVCAircraft"

if [[ ! -d "${AIRCRAFT_DIR}/images" ]]; then
    download_http \
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz" \
        "$AIRCRAFT_ARCHIVE"

    verify_file "$AIRCRAFT_ARCHIVE"

    rm -rf "$AIRCRAFT_TMP"
    mkdir -p "$AIRCRAFT_TMP"

    log "开始解压 FGVCAircraft"
    tar -xzf "$AIRCRAFT_ARCHIVE" -C "$AIRCRAFT_TMP"

    AIRCRAFT_DATA_DIR="$(
        find "$AIRCRAFT_TMP" \
            -type d \
            -path "*/fgvc-aircraft-2013b/data" \
            -print \
            -quit
    )"

    if [[ -z "$AIRCRAFT_DATA_DIR" ]]; then
        AIRCRAFT_DATA_DIR="$(
            find "$AIRCRAFT_TMP" \
                -type d \
                -name "data" \
                -print \
                -quit
        )"
    fi

    if [[ -z "$AIRCRAFT_DATA_DIR" ]]; then
        echo "解压后的目录结构："
        find "$AIRCRAFT_TMP" -maxdepth 4 -print
        die "没有找到 FGVCAircraft 的 data 目录"
    fi

    rm -rf "$AIRCRAFT_DIR"
    mv "$AIRCRAFT_DATA_DIR" "$AIRCRAFT_DIR"

    rm -rf "$AIRCRAFT_TMP"
else
    log "FGVCAircraft 已经解压，跳过"
fi

# ============================================================
# 检查目录和文件
# ============================================================

log "开始检查数据集目录结构"

[[ -d "${PETS_DIR}/images" ]] \
    || die "缺少 ${PETS_DIR}/images"

[[ -d "${PETS_DIR}/annotations" ]] \
    || die "缺少 ${PETS_DIR}/annotations"

verify_file "${PETS_DIR}/split_zhou_OxfordPets.json"

[[ -d "${FLOWERS_DIR}/jpg" ]] \
    || die "缺少 ${FLOWERS_DIR}/jpg"

verify_file "${FLOWERS_DIR}/imagelabels.mat"
verify_file "${FLOWERS_DIR}/cat_to_name.json"
verify_file "${FLOWERS_DIR}/split_zhou_OxfordFlowers.json"

[[ -d "${FOOD_DIR}/images" ]] \
    || die "缺少 ${FOOD_DIR}/images"

[[ -d "${FOOD_DIR}/meta" ]] \
    || die "缺少 ${FOOD_DIR}/meta"

verify_file "${FOOD_DIR}/split_zhou_Food101.json"

[[ -d "${AIRCRAFT_DIR}/images" ]] \
    || die "缺少 ${AIRCRAFT_DIR}/images"

log "四个数据集均已下载并解压完成"

echo
echo "最终目录结构："

for dataset_dir in \
    "$PETS_DIR" \
    "$FLOWERS_DIR" \
    "$FOOD_DIR" \
    "$AIRCRAFT_DIR"
do
    echo
    echo "===== $dataset_dir ====="
    find "$dataset_dir" -maxdepth 2 -print | sort
done