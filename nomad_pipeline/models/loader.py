"""Завантаження вагів (авто з Google Drive) і побудова моделей з visualnav-transformer."""
import os
import subprocess
import sys
import yaml
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TP = os.path.join(ROOT, "third_party")
VNT = os.path.join(TP, "visualnav-transformer")
DP = os.path.join(TP, "diffusion_policy")

THIRD_PARTY_REPOS = {
    VNT: "https://github.com/robodhruv/visualnav-transformer.git",
    DP: "https://github.com/real-stanford/diffusion_policy.git",
}


def ensure_third_party() -> None:
    """Автоматично клонує необхідні репозиторії у third_party/ при першому запуску."""
    os.makedirs(TP, exist_ok=True)
    for target_dir, repo_url in THIRD_PARTY_REPOS.items():
        if not os.path.exists(target_dir):
            print(f"[setup] Клонування {repo_url} -> {target_dir}...")
            subprocess.run(["git", "clone", "--depth", "1", repo_url, target_dir], check=True)
    for p in (DP, os.path.join(VNT, "train")):
        if p not in sys.path:
            sys.path.insert(0, p)


for p in (DP, os.path.join(VNT, "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

WEIGHTS_DIR = os.path.join(ROOT, "weights")
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg"
CKPT_NAMES = {"gnm": "gnm.pth", "vint": "vint.pth", "nomad": "nomad.pth"}
CONFIGS = {m: os.path.join(VNT, "train", "config", f"{m}.yaml") for m in CKPT_NAMES}


def ensure_weights(model: str, use_finetuned: bool = True) -> str:
    """Повертає шлях до .pth; якщо є дотреновані ваги для приміщень — використовує їх."""
    if model == "nomad" and use_finetuned:
        custom_w = os.environ.get("NOMAD_WEIGHTS_PATH")
        if custom_w and os.path.exists(custom_w):
            return custom_w
        for fname in ("nomad_tello_finetuned.pth", "nomad_indoor_finetuned.pth"):
            ft_path = os.path.join(WEIGHTS_DIR, fname)
            if os.path.exists(ft_path):
                return ft_path
    name = CKPT_NAMES[model]
    path = os.path.join(WEIGHTS_DIR, name)
    if os.path.exists(path):
        return path
    import gdown
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"[weights] {name} не знайдено, завантаження з Google Drive...")
    files = gdown.download_folder(DRIVE_FOLDER, output=os.path.join(WEIGHTS_DIR, "_tmp"),
                                  quiet=True, skip_download=True, use_cookies=False)
    match = [f for f in files if os.path.basename(f.path) == name]
    if not match:
        raise FileNotFoundError(f"{name} відсутній у папці Drive; покладіть вручну в {WEIGHTS_DIR}")
    gdown.download(id=match[0].id, output=path, quiet=False)
    if not os.path.exists(path):
        raise RuntimeError(f"Не вдалося завантажити {name} (ліміт Google Drive?). "
                           f"Завантажте вручну: {DRIVE_FOLDER}")
    return path


def load_config(model: str) -> dict:
    ensure_third_party()
    with open(CONFIGS[model]) as f:
        return yaml.safe_load(f)


def load_model(model_name: str, device, use_finetuned: bool = True):
    ensure_third_party()
    from vint_train.models.gnm.gnm import GNM
    from vint_train.models.vint.vint import ViNT
    from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
    from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    c = load_config(model_name)
    if model_name == "gnm":
        model = GNM(c["context_size"], c["len_traj_pred"], c["learn_angle"],
                    c["obs_encoding_size"], c["goal_encoding_size"])
    elif model_name == "vint":
        model = ViNT(context_size=c["context_size"], len_traj_pred=c["len_traj_pred"],
                     learn_angle=c["learn_angle"], obs_encoder=c["obs_encoder"],
                     obs_encoding_size=c["obs_encoding_size"], late_fusion=c.get("late_fusion", False),
                     mha_num_attention_heads=c["mha_num_attention_heads"],
                     mha_num_attention_layers=c["mha_num_attention_layers"],
                     mha_ff_dim_factor=c["mha_ff_dim_factor"])
    else:
        enc = NoMaD_ViNT(obs_encoding_size=c["encoding_size"], context_size=c["context_size"],
                         mha_num_attention_heads=c["mha_num_attention_heads"],
                         mha_num_attention_layers=c["mha_num_attention_layers"],
                         mha_ff_dim_factor=c["mha_ff_dim_factor"])
        enc = replace_bn_with_gn(enc)
        unet = ConditionalUnet1D(input_dim=2, global_cond_dim=c["encoding_size"],
                                 down_dims=c["down_dims"], cond_predict_scale=c["cond_predict_scale"])
        model = NoMaD(vision_encoder=enc, noise_pred_net=unet,
                      dist_pred_net=DenseNetwork(embedding_dim=c["encoding_size"]))

    ckpt = torch.load(ensure_weights(model_name, use_finetuned=use_finetuned), map_location=device, weights_only=False)
    if model_name == "nomad":
        model.load_state_dict(ckpt, strict=False)
    else:
        m = ckpt["model"]
        sd = m.module.state_dict() if hasattr(m, "module") else m.state_dict()
        model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), c
