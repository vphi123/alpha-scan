import json
from pathlib import Path

nb_path = Path("AlphaScan_Plan2A_Colab.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

yaml_code = [
    "# Ensure model_config.yaml is present\n",
    "cfg_yaml = '''TRAIN:\n",
    "  USE_FP16: true\n",
    "  FP16_TYPE: bfloat16\n",
    "MODEL:\n",
    "  IMAGE_SIZE:\n",
    "  - 512\n",
    "  - 512\n",
    "  IMAGE_MEAN:\n",
    "  - 0.485\n",
    "  - 0.456\n",
    "  - 0.406\n",
    "  IMAGE_STD:\n",
    "  - 0.229\n",
    "  - 0.224\n",
    "  - 0.225\n",
    "  BACKBONE:\n",
    "    TYPE: dinov3_vith16plus\n",
    "    PRETRAINED_WEIGHTS: ''\n",
    "    FROZEN_STAGES: -1\n",
    "    DROP_PATH_RATE: 0.1\n",
    "  DECODER:\n",
    "    TYPE: sam\n",
    "    DIM: 1024\n",
    "    DEPTH: 6\n",
    "    HEADS: 8\n",
    "    MLP_DIM: 1024\n",
    "    DIM_HEAD: 64\n",
    "    LAYER_SCALE_INIT: 0.0\n",
    "    DROP_RATE: 0.0\n",
    "    ATTN_DROP_RATE: 0.0\n",
    "    DROP_PATH_RATE: 0.0\n",
    "    FFN_TYPE: origin\n",
    "    ENABLE_TWOWAY: false\n",
    "    REPEAT_PE: true\n",
    "    FROZEN: false\n",
    "    CONDITION_TYPE: cliff\n",
    "    USE_INTRIN_CENTER: true\n",
    "    DO_INTERM_PREDS: true\n",
    "    DO_INTERM_SUP: true\n",
    "    DO_KEYPOINT_TOKENS: true\n",
    "    DO_HAND_DETECT_TOKENS: true\n",
    "    KEYPOINT_TOKEN_UPDATE: v2\n",
    "    KEYPOINT_TOKEN_UPDATE_COORD_EMB_USE_MLP: true\n",
    "    DO_KEYPOINT3D_TOKENS: true\n",
    "  PROMPT_ENCODER:\n",
    "    ENABLE: true\n",
    "    MAX_NUM_CLICKS: 2\n",
    "    PROMPT_KEYPOINTS: mhr70\n",
    "    FROZEN: false\n",
    "    KEYPOINT_SAMPLER:\n",
    "      TYPE: v1\n",
    "      WORST_RATIO: 0.8\n",
    "      KEYBODY_RATIO: 0.8\n",
    "      NEGATIVE_RATIO: 0.1\n",
    "      DUMMY_RATIO: 0.1\n",
    "      DISTANCE_THRESH: 0.0001\n",
    "    MASK_EMBED_TYPE: v2\n",
    "    MASK_PROMPT: v1\n",
    "  PERSON_HEAD:\n",
    "    POSE_TYPE: mhr\n",
    "    CAMERA_ENABLE: true\n",
    "    CAMERA_TYPE: perspective\n",
    "    ZERO_POSE_INIT: true\n",
    "    ZERO_POSE_INIT_BODY_FACTOR: 1\n",
    "    MHR_HEAD:\n",
    "      MLP_DEPTH: 2\n",
    "      MLP_CHANNEL_DIV_FACTOR: 1\n",
    "    CAMERA_HEAD:\n",
    "      MLP_DEPTH: 2\n",
    "      MLP_CHANNEL_DIV_FACTOR: 1\n",
    "  DEFAULT_SCALE_FACTOR_HAND: 10\n",
    "  ENABLE_BODY: true\n",
    "  ENABLE_HAND: true\n",
    "'''\n",
    "for p in [REPO_DIR / 'checkpoints', LOCAL_CKPT_DIR, Path('/content/sam-3d-body/checkpoints')]:\n",
    "    p.mkdir(parents=True, exist_ok=True)\n",
    "    (p / 'model_config.yaml').write_text(cfg_yaml, encoding='utf-8')\n",
]

for cell in nb["cells"]:
    src = cell.get("source", [])
    src_str = "".join(src)
    # Cell 2: add braceexpand, webdataset
    if "Install trimesh, xatlas" in src_str:
        new_src = []
        for line in src:
            new_src.append(line)
            if '"xatlas",' in line:
                new_src.append('    "braceexpand",\n')
                new_src.append('    "webdataset",\n')
        cell["source"] = new_src

    # Cell 4: add model_config.yaml writing
    if "LOCAL_CKPT_DIR = REPO_DIR / \"checkpoints\" / \"sam-3d-body-dinov3\"" in src_str:
        new_src = []
        for line in src:
            new_src.append(line)
            if "LOCAL_CKPT_DIR.parent.mkdir" in line:
                new_src.extend(yaml_code)
        cell["source"] = new_src

    # Cell 7b (Fast restart): ensure braceexpand and model_config.yaml
    if "⚡ 1-Cell Fast Restart" in src_str or "Fast-restarting API" in src_str:
        new_src = []
        injected = False
        for line in src:
            if not injected and ("api_cmd = [" in line or "api_proc = subprocess.Popen" in line):
                new_src.append("subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'braceexpand', 'webdataset'], check=False)\n")
                new_src.extend(yaml_code)
                injected = True
            new_src.append(line)
        cell["source"] = new_src

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Updated AlphaScan_Plan2A_Colab.ipynb successfully.")
