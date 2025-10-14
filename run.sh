set -euo pipefail

MODEL="${1:-ALIGNREC_ANCHOR}"    
DATASET="baby"
EXTRA_ARGS=()

shift 1 || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|-d)
      DATASET="$2"; shift 2;;
    *)
      EXTRA_ARGS+=("$1"); shift 1;;
  esac
done

python src/main.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --log_file_name "${MODEL}" \
  --ui_cosine_loss \
  --multimodal_data_dir "data/${DATASET}_beit3_128token_add_title_brand_to_text/" \
  "${EXTRA_ARGS[@]}"