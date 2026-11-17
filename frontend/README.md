# Interlock validation studio

A local, buildless viewer for actual validation solves. Uses the repository's Python environment; no Node dependencies or build step.

From the repository root:

```sh
.venv/bin/python frontend/server.py --port 8765
```

Open <http://127.0.0.1:8765>. The server binds only to loopback. Apple MPS is used when available, otherwise CPU. Keep the server running while using the viewer.

Choose a model and a category, then load one of 100 validation images. Flat and slanted cuts support anchored and unanchored modes. Playback supports pause, rewind, single-step, timeline scrubbing, and speed changes. Keyboard shortcuts: Space to play/pause; Left/Right to step when focus is outside a control.

Straight and slanted cuts now offer three **Assembly mode** choices: **Unanchored**, **Top-left anchored**, and **Strongest pair**. Strongest pair finds the global highest raw directed pair score (excluding self-pairs) and commits that pair on an open integer grid. It then repeatedly commits the best unused piece/frontier-slot combination, ranked by the mean raw score against every occupied neighbor touching that slot. It never receives the image layout, board dimensions, or true piece positions. Pieces remain upright, no piece is reused, holes and nonrectangular shapes are allowed, and placements are not subsequently revised. This is a separate greedy growth experiment, not the stage-2/3/4 rectangular solver with a different seed.

The replay starts with the pair highlighted, then adds one piece per step. Board extents show the final bounding box for a stable playback viewport, not a boundary constraint. All metrics describe the completed growth run. **Aligned location accuracy** is the fraction of pieces agreeing with the ground truth after choosing the single best global integer translation (no rotation, reflection, or per-piece adjustment). Ground truth is used only after assembly to evaluate it. **Neighbor precision** is correct formed joins / all formed joins; **neighbor recall** is recovered true joins / all true joins, so a sparse shape cannot earn a misleadingly strong score by avoiding joins. Pieces placed and assembly bounds are shown separately. Full completion requires every piece to match under one translation. Strongest-pair failures have a separate evaluated-case list and reuse the same CNN logits as the other modes.

Use **Try an image** for a custom experiment: choose regular jigsaw, straight, or slanted cuts, a 4×4 / 6×6 / 8×8 / 10×10 grid, and upload an image or choose one of 12 small gallery previews. Uploads stay under the local ignored cache. Images are oriented using EXIF, center-cropped to a square and resized to approximately 400×400. Uploads are limited to 12 MB, 40 megapixels, and at least 120 pixels per side. Custom results have known ground truth from the generated cuts, but are labeled separately from held-out validation. Fixed cut/shuffle seeds make model and anchor comparisons repeatable.

In validation mode, **Unsolved** opens clickable failed cases with image numbers and final location accuracy. **Next unsolved** jumps to the next known failure and wraps at the end. Results are scoped to the selected checkpoint, category, and anchor mode, with explicit evaluated coverage out of 100. The list includes matching saved 30-puzzle anchored-experiment results and current replay caches; custom-image experiments are excluded. Untested puzzles are never treated as solved. Loading more validation images updates the list. Cut selection in custom mode lives only in the main toolbar; validation category navigation is hidden there.

At the final replay step, incorrectly placed pieces receive a light red overlay and red cell boundary. **Mark errors** toggles it. In strongest-pair mode the error mask uses the recorded global translation alignment, matching the displayed location metric. Empty cells are not counted as misplaced pieces; the legend shows the actual number of wrong pieces. Earlier playback steps remain unmarked so provisional guesses are not confused with final errors.

## Complete validation cache

With the viewer server running, start or resume the full evaluation:

```sh
.venv/bin/python frontend/precompute.py
```

This evaluates all 100 held-out images for every available model and all eight categories. Straight and slanted cuts additionally run all three assembly modes. With the three bundled checkpoints that is **3,600 replay results and up to 2,400 unique CNN scoring runs**. Scores are shared among modes. The batch uses the server's single inference queue, so interactive requests and batch jobs do not run concurrent GPU calls. Images are visited in matching index order across models. Runtime can be several hours, depending on hardware, existing caches, and interactive use; keep the server and batch process running.

The run resumes by reusing completed replay files and score caches. Stop the batch with Ctrl-C, then rerun the same command to resume. A file lock prevents duplicate batches. Failed requests are recorded; a run with any errors is marked incomplete. Resume retries missing cases. `--count 1` checks one image per combination; `--url` selects a different viewer port.

Progress and per-model/category/mode aggregates are saved atomically under the ignored `artifacts/viewer_cache/` directory:

- `batch-status.json`: progress, current case, errors, checkpoint/code fingerprints, completion state.
- `batch-summary.json`: evaluated count, full-puzzle solve rate, mean piece-location and neighbor accuracy, and zero-based failed image indices for each combination. Strongest-pair location accuracy is translation aligned.
- Each replay folder contains `result.json`, `pieces.png`, and `reference.png`; reopening cached results requires no inference.

The viewer polls batch progress and refreshes failure counts while it runs. **Unsolved F / E tested** means F failures out of E evaluated images, not a complete 100-image result until E=100. Freshly loaded cases used to increase this count because evaluation was on demand. Results are never generalized from tested cases to untested ones. Custom-image uploads are excluded from the matched validation batch.

**Notify when done** requests browser notification permission. Leave the viewer open: when it observes a completed batch it updates the page title, announces completion, and sends a browser notification if permitted. Without permission the in-page completion banner remains available. This local process does not send a later chat message or email.

## Evaluating the updater

For rectangular solver modes, placement accuracy, neighbor accuracy, and completion state now follow the selected playback step. CNN top-1 remains fixed. The live updater strip reports correct pieces, change from the last stage-2 placement in percentage points, and incorrectly locked pieces. At final assignment, wrong-lock counts retain the stage-3 lock mask so new Hungarian assignments are not mislabeled as updater locks.

The validation **Does refinement help?** chart shows total full-puzzle solve rate and mean location accuracy after the last stage-2 placement, each stage-3 refinement round, and stage-4 assignment. A scrubber-linked cursor follows the current puzzle's stage. Its table includes exact solved counts, placement/neighbor accuracy, and wrong-lock totals. Counts of puzzles recovered and initially solved puzzles broken compare initial placement with final assignment.

Every point uses the same checkpoint/category/anchor cohort with actual replay snapshots; historical final-only summaries are excluded. If a puzzle stops refinement before the cohort's longest run, its last stage-3 state is carried forward, and its stage-4 assignment is applied only at the separate final point. This avoids changing the denominator or crediting early Hungarian assignment as later refinement. Coverage is explicit and grows as the batch fills. Analysis is derived from existing traces, requires no CNN work, and does not invalidate inference caches. Custom images show live metrics but no validation-cohort chart. Strongest-pair growth has no multistage updater and retains its separate final aligned metrics.

## Models and adding weights

| Menu entry | Actual checkpoint | Provenance |
|---|---|---|
| Robust comparator | `checkpoints/robust_comparator_hf.pt` | Published `dm157/interlock-edge-comparator` model, epoch 15, trained on 1,500 SA-1B images with outline errors, transplant negatives, and straight/slanted training examples. Used in the original robustness study and our anchored comparison. |
| Original local comparator | `checkpoints/edge_comparator.pt` | Local epoch-2 checkpoint for the ordinary comparator. Its metadata does not record the full training configuration; it is not identified as the published study's baseline. |
| Single-tab comparator | `checkpoints/edge_comparator_single_tab_full.pt` | Local epoch-4 checkpoint with `single_tab_shape=True`, trained for the experiment using a canonical internal tab curve. |

The menu maps IDs to real checkpoint paths in `server.py`. The server calls `load_comparator()` to rebuild `EdgeComparator` from `config`, load its `state_dict`, score neighboring pieces, and feed those scores into the solver. No accuracy values are mocked.

To add your own compatible model without editing the registry, restart with:

```sh
.venv/bin/python frontend/server.py --port 8765 --checkpoint "My model=/absolute/path/to/model.pt"
```

Repeat `--checkpoint` to add more than one. The checkpoint is validated before the server starts. It must use the repository's `EdgeComparator` architecture and input format. Save a trained model in the expected format with:

```python
from interlock.models.edge_comparator import save_comparator
save_comparator(model, "/absolute/path/to/model.pt")
```

This writes `config` and `state_dict`. A bare PyTorch state dict needs to be loaded into a compatible `EdgeComparator` and saved with this helper first. Different model architectures require a loader/scoring adapter. Restart after replacing weights in an existing file; hashes ensure a new checkpoint gets separate cached results.

## Data and reproducibility

- Image 1–100 maps to SA-1B validation indices 100–199, in `data/processed/manifest.csv` order. The processed images must exist locally.
- Flat cuts, perturbations, and scoring use `experiment.py`, copied from `testing/robust_comparator.py` on the `greedy` branch. This intentionally retains its rasterized flat-cut generator, which differs from `cut(straight_edges=True)`.
- Cut seed = validation index. Shuffle seed = 10000 + zero-based viewer image index. Perturbation seed = `[10000, condition index, viewer image index]`, matching the published robust evaluation.
- `Robust comparator` uses `checkpoints/robust_comparator_hf.pt`. `Original local comparator` uses `checkpoints/edge_comparator.pt`; this is not asserted to be the baseline checkpoint from the published robustness study. `Single-tab comparator` uses `checkpoints/edge_comparator_single_tab_full.pt`. Only existing checkpoints are offered. Add models to `MODELS` in `server.py`.
- The original published flat-cut comparison used only images 1–30. Images 31–100 are additional on-demand evaluations, explicitly labeled in the interface. No cohort-wide scores are inferred from a single replay.
- The first 30 robust flat-cut logits from `artifacts/anchored_flat/` are reused only with matching manifest and checkpoint hashes. Other logits and replay data live under `artifacts/viewer_cache/`, keyed by inference code, manifest, checkpoint, category, and image. UI and routing changes do not invalidate inference caches. Anchor modes reuse scores. One inference job runs at a time.
- First loads perform real inference and can take a minute on large puzzles. Replays are subsequently cached. Changing categories during scoring queues a new job; it does not interrupt an already running GPU call.
- This is a solver-state replay, not a stream of wall-clock model execution. Metrics always describe the final assembly. The reference image is for inspection; anchored mode supplies only the true top-left piece ID.

## Existing visualizers

`scripts/visualize_solve.py` generates a static 2×3 Matplotlib figure: original, shuffled pieces, stage-2 fill-order heatmap, stage-2 placement, final placement, and final confidence. Red rectangles mark wrong slots; dots on the confidence plot indicate stage-3 locks. It solves one puzzle using the selected checkpoint and supports an optional known top-left piece, straight grids, noise, and a fixed tab shape. It does not animate or persist an experiment catalog.

`scripts/animate_solve.py` requests `Snapshot` records from the solver. Stage 2 records individual slot guesses, stage 3 records refinement rounds, and stage 4 records final assignment. Muted pieces are provisional; committed pieces appear in full color and fade from the shuffled tray. The script plays in Matplotlib or writes a GIF with repeated frames to linger on refinement/final states.

This UI uses the same solver trace mechanism and draws transparent piece sprites on a canvas. It retains every solver snapshot so you can seek backward without rerunning inference. Stage-2 guesses can repeat pieces; the final assignment is a permutation.

## Verification

With the server on port 8765, `PYTHONPATH=. .venv/bin/python frontend/check_api.py` exercises all eight categories, all three assembly modes for flat cuts, image 40, all bundled models, gallery thumbnails, real uploads, all three custom cuts, invalid input handling, asset loading, final-piece uniqueness, anchor preservation, and independently recomputed free-assembly alignment metrics. This performs real inference for uncached cases. `tests/test_strongest_pair.py` checks interior seeding, outward growth, conflicting-neighbor scoring, translation alignment, adjacency recall, ties, and invalid inputs.

The page uses Google Fonts for DM Sans and Special Elite with system sans-serif and Courier fallbacks. All puzzle data and model inference stay local.
