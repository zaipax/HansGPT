#!/usr/bin/env bash
set -euo pipefail
cd /root/HansGPT
export PATH="/root/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
mode=${1:?smoke or full}
[[ "$mode" == smoke || "$mode" == full ]] || exit 2
name=chinese_multidomain_v1
[[ "$mode" == smoke ]] && name=chinese_multidomain_smoke_v4
workers=24
[[ "$mode" == smoke ]] && workers=4
[[ -z "$(git status --porcelain)" ]] || exit 2
if [[ "$mode" == full ]]; then
  uv run --frozen python - <<'PY'
import json,subprocess
from pathlib import Path
from hansgpt_research.prepare_corpus import digest_file
p=Path('data/processed/chinese_multidomain_smoke_v4')
r=json.loads((p/'verification.json').read_text())
m=json.loads((p/'manifest.json').read_text())
assert r['passed'] and r['manifest_sha256']==digest_file(p/'manifest.json')
assert m['source_config_sha256']==digest_file(Path('configs/datasets/chinese_multidomain_v1.json'))
assert m['git_commit']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert all(m['family_han'].get(name,0)>0 for name in ('literature','finance','medicine','education_web'))
PY
fi
uv run --frozen python scripts/prepare_multidomain_corpus.py --mode "$mode" \
  --workers "$workers" \
  --output "data/processed/$name" --interim "data/interim/$name"
uv run --frozen python scripts/verify_multidomain_corpus.py "data/processed/$name"
uv run --frozen python - "$name" <<'PY'
import json,sys
from pathlib import Path
from hansgpt_research.prepare_corpus import digest_file
name=sys.argv[1]
root=Path('data/processed')/name
state={'phase':'complete','verification_sha256':digest_file(root/'verification.json'),'output':str(root)}
path=Path('data/interim')/name/'status.json'
path.write_text(json.dumps(state,indent=2)+'\n')
PY
