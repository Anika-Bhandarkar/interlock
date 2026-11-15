"""Package a trained comparator for the Hugging Face Hub, and upload it when asked.

Builds a release folder (default checkpoints/hf_release/, gitignored) with:
- edge_comparator.pt: the checkpoint as saved, loadable with interlock.models.edge_comparator.load_comparator;
- config.json: architecture, input shape, the fitted temperatures and the training settings;
- robust_eval.md: the full evaluation tables;
- README.md: a model card written from testing/results/robust_eval.json, so every number in it is measured.

Nothing leaves this machine unless --upload is given. Uploading needs a Hugging Face login with write access
(run `hf auth login` once) and the huggingface_hub package, which isn't a project dependency:

    uv run python testing/publish_hf.py                                   # build and review the folder
    uv run --with huggingface_hub python testing/publish_hf.py --repo USER/interlock-edge-comparator --upload

--private creates the repo private (you can make it public later on the Hub).
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_URL = "https://github.com/Anika-Bhandarkar/interlock"
SA1B_LICENSE_URL = "https://ai.meta.com/datasets/segment-anything-downloads/"
DATASETS = {
    "SA-1B": f"Meta's [SA-1B]({SA1B_LICENSE_URL})",
    "Imagenette": "[Imagenette](https://github.com/fastai/imagenette), a 10-class ImageNet subset",
}
TEST_SETS = {
    "clean": "Clean pieces",
    "pixel_noise": "Pixel noise (σ = 12)",
    "shrink_1px_half": "Half the outlines shrunk 1 px",
    "uneven_outline": "Uneven outline error (≤ 5% of a piece)",
    "smoothed_outline": "Smoothed outlines",
    "flat_even": "Flat edges, even grid",
    "flat_diagonal": "Flat edges, diagonal cuts",
    "color_removed": "Color removed (diagnostic: should drop)",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def results_table(results: dict, model: str, baseline: str | None) -> list[str]:
    """Markdown rows: comparator top-1 and solver results per test set, next to the baseline's if there is one."""
    names = [baseline, model] if baseline else [model]
    header = "| Test set | " + " | ".join(f"Top-1, {n}" for n in names) + " | " + \
             " | ".join(f"Placed / neighbors / solved, {n}" for n in names) + " |"
    lines = [header, "|---|" + "---|" * (2 * len(names))]
    for key, label in TEST_SETS.items():
        if key not in results[model]["conditions"]:
            continue
        sets = [results[n]["conditions"][key] for n in names]
        top1 = " | ".join(pct(s["top1"]) for s in sets)
        solver = " | ".join(f"{pct(s['solver']['stock']['direct'])} / {pct(s['solver']['stock']['neighbors'])} / "
                            f"{pct(s['solver']['stock']['perfect'])}" for s in sets)
        lines.append(f"| {label} | {top1} | {solver} |")
    return lines


def model_card(repo: str, config: dict, results: dict, model: str, baseline: str | None, license_id: str) -> str:
    training = config["training"]
    t_edge = config["temperature"]["T_edge"]
    sets = results[model]["conditions"]
    pieces = sets["clean"]["pieces_per_puzzle"]
    sa1b = training.get("dataset") == "SA-1B"
    license_meta = [f"license: {license_id}"]
    if sa1b and license_id == "other":
        license_meta += ["license_name: sa-1b-dataset-research-license", f"license_link: {SA1B_LICENSE_URL}"]
    lines = [
        "---",
        *license_meta,
        "library_name: pytorch",
        "tags: [jigsaw-puzzle, computer-vision, edge-matching]",
        "---",
        "",
        f"# {repo.split('/')[-1]}",
        "",
        "Edge comparator from [Interlock](" + CODE_URL + "), an automatic jigsaw puzzle solver. Given two pieces, it "
        "scores how likely the second is the first one's right-hand neighbor; rotating both pieces covers the other "
        "three directions. The solver turns these scores into a full placement.",
        "",
        "This version is trained to hold up when piece outlines aren't pixel-perfect, as they won't be when pieces "
        "come from a photo, and to match on the picture as well as the shape.",
        "",
        "## Input and output",
        f"- Input: seam windows of shape (5, {config['input_shape'][0]}, {config['input_shape'][1]}), uint8: RGB plus "
        "each piece's mask, cropped around the seam and zero-padded to that size (`pairs.seam_windows`, `pairs.pad_to`).",
        "- Output: one logit per window. Higher means more likely neighbors.",
        f"- Calibration: divide logits by **T = {t_edge:.2f}** before a softmax over candidates "
        "(`testing/robust_comparator.scale_logits` does this for the solver).",
        "",
        "## Use",
        "```python",
        "from huggingface_hub import hf_hub_download",
        "from interlock.models.edge_comparator import load_comparator, pick_device, score_matrix",
        "",
        f'model = load_comparator(hf_hub_download("{repo}", "edge_comparator.pt"), pick_device())',
        "logits = score_matrix(model, pieces, masks)  # (4, N, N): right, below, left, above",
        "```",
        "`pieces` and `masks` are what `interlock.data_collection.cutting.cut` returns. The code is in the "
        f"[Interlock repository]({CODE_URL}) (`testing/` on the `greedy` branch).",
        "",
        "## Training",
        f"- Data: {training['num_train_images']} images from {DATASETS.get(training.get('dataset'), 'an unrecorded image set')}, "
        "resized and cropped to 400 × 400, each cut into a new random puzzle every epoch (30-100 px pieces, 6 tab "
        f"shapes per puzzle). Manifest hash `{training['manifest_sha1']}`.",
        "- Every piece's outline gets segmentation-style error: it grows or shrinks by up to 3.5% of a piece, "
        "unevenly, capped at 5%; 20% of pieces stay exact.",
        f"- {pct(training['transplant_share'])} of wrong examples are decoys with the true neighbor's exact outline "
        "but another part of the picture, so only color can reject them.",
        f"- {pct(training['flat_even_share'])} of puzzles are cut on an even grid of straight lines and "
        f"{pct(training['flat_diagonal_share'])} with slanted straight lines, so edge length and angle can't decide a "
        "match either.",
        f"- {training['epochs']} epoch{'s' if training['epochs'] != 1 else ''}, AdamW, one-cycle learning rate up to "
        f"{training['lr']}, on Apple Silicon (MPS). Best epoch ({config['epoch']}) picked on {training['eval_puzzles']} "
        "dev puzzles with outline error.",
        "",
        "## Evaluation",
        f"{sets['clean']['puzzles']} held-out puzzles per test set ({sets.get('flat_even', sets['clean'])['puzzles']} for "
        f"flat edges), val images 100 onward, about {pieces:.0f} pieces each, never used for training or tuning. "
        "\"Placed / neighbors / solved\" is the share of pieces in the right space, of neighboring pairs that are true "
        "neighbors, and of puzzles fully solved, by Interlock's greedy solver (stages 2-4, stock settings). Flat-edged "
        "puzzles have no corner or edge shapes to anchor the grid, so their assemblies can come out shifted: read "
        "neighbors there.",
        "",
        *results_table(results, model, baseline),
        "",
        "Full tables, including calibration, are in `robust_eval.md`.",
        "",
        "## Limitations",
        "- Trained and tested on synthetic cuts of real images, not on photos of physical pieces.",
        f"- {sets['uneven_outline']['ruled_out']} of {sets['uneven_outline']['pairs']} true pairs under outline error "
        "are ruled out before the comparator runs, because `pairs.side_types` misreads a tab or blank.",
        "- The greedy solver locks most pieces early (its threshold 1), so better scores don't always mean better "
        "placements.",
        "",
        "## License",
        *([
            "**Research and non-commercial use only.** These weights were trained on Meta's SA-1B dataset, which is "
            f"released under the [SA-1B Dataset Research License]({SA1B_LICENSE_URL}). That license allows research "
            "on a non-commercial basis and forbids any commercial or production use, surveillance, biometric "
            "processing, and attempts to identify people in the images. Use these weights under the same terms. The "
            "release contains no SA-1B images or masks.",
        ] if sa1b else [
            "Check the training images' license before using these weights commercially.",
        ]),
        "",
    ]
    return "\n".join(lines)


def build(args) -> Path:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "training" not in checkpoint:
        raise SystemExit(f"{args.checkpoint} has no training settings; train it with testing/train_robust_comparator.py")
    results = json.loads(args.results.read_text())
    if args.model not in results:
        raise SystemExit(f"{args.results} has no results for {args.model!r}; run testing/eval_robust.py first")
    baseline = args.baseline if args.baseline in results else None

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(args.checkpoint, out / "edge_comparator.pt")
    shutil.copy(args.results.with_suffix(".md"), out / "robust_eval.md")
    config = {
        **checkpoint["config"],
        "architecture": "interlock.models.edge_comparator.EdgeComparator",
        "epoch": checkpoint["epoch"],
        "temperature": {"T_edge": results[args.model]["T_edge"], "T_space": results[args.model]["T_space"]},
        "training": checkpoint["training"],
        "dev_metrics": checkpoint.get("dev_metrics", {}),
    }
    (out / "config.json").write_text(json.dumps(config, indent=1, default=list) + "\n")
    (out / "README.md").write_text(model_card(args.repo, config, results, args.model, baseline, args.license))
    return out


def upload(folder: Path, repo: str, private: bool) -> str:
    from huggingface_hub import HfApi  # only needed here; run with `uv run --with huggingface_hub`

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo, repo_type="model", commit_message="Upload Interlock edge comparator")
    return f"https://huggingface.co/{repo}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="interlock/interlock-edge-comparator", help="Hub repo id, USER/NAME")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "robust_comparator.pt")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "testing" / "results" / "robust_eval.json")
    parser.add_argument("--model", default="robust", help="name of this model in the results")
    parser.add_argument("--baseline", default="baseline", help="name of the model to compare against in the card")
    parser.add_argument("--license", default="other", help="license id for the card's metadata")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "hf_release")
    parser.add_argument("--upload", action="store_true", help="upload to the Hub (otherwise only build the folder)")
    parser.add_argument("--private", action="store_true", help="create the Hub repo as private")
    args = parser.parse_args()

    folder = build(args)
    print(f"built {folder}: " + ", ".join(sorted(p.name for p in folder.iterdir())))
    if args.upload:
        print(f"uploaded to {upload(folder, args.repo, args.private)}")
    else:
        print("not uploaded; review the folder, then re-run with --repo USER/NAME --upload")


if __name__ == "__main__":
    main()
