#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd "$project_dir/.." && pwd)"
data_dir="${1:?Usage: run_raw_session.sh DATA_DIR [POSITION] [SESSION_ID] [OUTPUT_DIR]}"
position="${2:-pos01}"
default_session_id="$(basename "$data_dir")_${position}"
session_id="${3:-$default_session_id}"
output_dir="${4:-$repo_dir/data/towel_test/labeled_sets/$session_id}"

clean_reference="$data_dir/clean_${position}_ref_001"
clean_control="$data_dir/clean_${position}_control_001"

if [[ ! -e "$clean_reference" ]]; then
  echo "Missing clean reference: $clean_reference" >&2
  exit 2
fi
if [[ ! -e "$clean_control" ]]; then
  echo "Missing independent clean control: $clean_control" >&2
  exit 2
fi

shopt -s nullglob
smoke_bags=("$data_dir"/smoke_"$position"_*)
shopt -u nullglob
if (( ${#smoke_bags[@]} == 0 )); then
  echo "No smoky recordings match $data_dir/smoke_${position}_*" >&2
  exit 2
fi

args=(
  dataset
  --config "$project_dir/config/raw_dataset.toml"
  --clean-reference "$clean_reference"
  --clean-control "$clean_control"
  --output "$output_dir"
  --topic /livox/lidar
  --session-id "$session_id"
)
for bag in "${smoke_bags[@]}"; do
  args+=(--smoke "$bag")
done

"$project_dir/.venv/bin/smoke-label" "${args[@]}"
