# Image generation

The hub can generate images through `POST /v1/images/generations` (OpenAI
Images shape). This note records *what backend actually does the work* and
*why the contract looks the way it does* — both differ from the obvious
first guess.

There are **two** image backends, picked by the `model` field. They are
additive, not a migration: which one wins long-term is a call to make once
there's real usage to judge by (#492).

| Model id | Backend | Where it runs | Editing | Typical latency |
|---|---|---|---|---|
| `gemini_image` | Google Imagen via the `agy` CLI | Google's servers, on the AI Pro/Ultra subscription | yes (procedural, slow) | seconds |
| `flux1_local` (alias `flux`) | FLUX.1 [dev] fp8 via ComfyUI | tower's RTX 5060 Ti, fully local | no | ~40 s at 1024², ~96 s at 4K |
| `flux2_klein` (alias `klein`) | FLUX.2 [klein] 4B fp8 via ComfyUI | same GPU | no | **~25 s** at 1024² |
| `flux2_local` (alias `flux2`) | FLUX.2 [dev] 32B Q4_K_M via ComfyUI | same GPU | no | **~10 min** at 1024² |

**Which to reach for:** `flux2_klein` is the everyday local choice — fastest of the three and genuinely good. `flux1_local` is the middle option and the only one with the 4K/upscale path exercised end to end. `flux2_local` is the quality ceiling this hardware can reach *at all*, at roughly 24x klein's cost; treat it as a "when it matters" model, not a default.

The rest of this note covers the Imagen path first (it came first, and its
constraints are the surprising ones), then the local FLUX path.

## What `agy` actually exposes (the spike finding)

Issue #114 set out to wire **Nano Banana / Nano Banana Pro** (Gemini image
models) into the hub via the Antigravity CLI (`agy`). The feasibility spike
found that premise is **not reachable**:

- `agy`'s `/model` picker offers **only text models** — as of `agy`
  1.1.8 that's Gemini 3.6 Flash, Gemini 3.5 Flash, Gemini 3.1 Pro,
  Claude Sonnet 4.6 (Thinking), Claude Opus 4.6 (Thinking), GPT-OSS 120B
  (Medium), with effort now a separate Low/Medium/High slider shared
  across whichever model is selected rather than baked into each row
  label (#440, #442). **There is no Nano Banana / image entry.**
- `agy` *can* generate images, but through its **agentic tool harness**: its
  only image backend is Google **Imagen**, reachable from inside any ordinary
  Gemini text session. Asked directly for "Nano Banana Pro", `agy` replied
  *"I used the built-in image generation model (Imagen)… Nano Banana Pro is
  not available to me."*
- The selected **text** model (Flash vs Pro) does **not** change the image
  model — both route to the same Imagen tool. So there is exactly **one**
  honest image id, `gemini_image`; a `_pro` sibling would be a second name
  for an identical backend.

## How the hub drives it

`src/gemini_cli.py::call_gemini_image` hosts the Imagen tool inside the
cheapest/fastest text session (`_IMAGE_HOST_MODEL = "Gemini 3.6 Flash"`,
kept in sync with `gemini_flash`'s `display_name`), under the same global
`_LOCK` + persisted-model-switch contract as text calls. It runs an
`agy -p` print-mode prompt that asks the model to
generate the image and **save it into a throwaway working dir**, then captures
whatever artifact lands there.

The artifact is identified by **magic bytes, never by file extension**: the
spike observed `agy` saving JPEG bytes under a `.png` name (it autonomously
ran a .NET `System.Drawing` conversion step), so trusting the name would
mislabel the media type. See `_sniff_image_media_type` / `_collect_image_artifact`.

## Editing (`POST /v1/images/edits`)

You can also **edit** an image: POST the image plus instructions (OpenAI
`/v1/images/edits` multipart shape) and get an edited image back. Internally
the upload is handed to `agy` as an `@<basename>` reference and the model is
asked to edit it.

Two honest caveats, both surfaced in the Playground UI:

- **It is not Imagen generative editing.** `agy` typically performs the edit
  by *writing image-processing code* (e.g. a Pillow/NumPy HSV transform) — it
  is agentic and procedural. Simple edits (recolor, crop, filter) come out
  well; complex semantic edits ("add a hat") are unreliable.
- **It is slow** — minutes, not seconds (a color swap measured ~4 min). The
  edit path uses a longer default timeout (`call_gemini_image` → 600 s) and the
  Playground proxy a 900 s client timeout.

## Local generation: FLUX.1 [dev] via ComfyUI (#492)

`flux1_local` runs entirely on tower's GPU — no subscription, no network call.

### Why FLUX.1 and not FLUX.2

Issue #492 originally specified **FLUX.2 [dev]** at "FP16 without
quantization", on the reading that a 16 GB card could hold it. That premise
does not survive contact with the numbers: FLUX.2 [dev] is a **32B** transformer
(~64 GB at bf16) and pairs with a **Mistral-Small-3.2-24B** text encoder. FP16 on
16 GB is not close to possible; it would need aggressive quantization *and*
weight offload — the very thing the issue scoped out as a Gaming-PC-only
concern.

**FLUX.1 [dev]** (12B) at fp8 is the honest fit for this GPU. It also proves the
plumbing: adding a quantized FLUX.2 row later is a `config/models.yaml` change
plus weights, not a rewrite, because everything below is model-agnostic.

The weights are the **all-in-one fp8 checkpoint**
(`Comfy-Org/flux1-dev` → `flux1-dev-fp8.safetensors`, 17.25 GB, ungated):
transformer + CLIP-L + T5-XXL + VAE in one file. That is why the workflow needs
a single `CheckpointLoaderSimple` and there are no separate text-encoder/VAE
downloads to keep in sync.

### How it's wired

Unlike `gemini_image` — a subscription path with no local process — ComfyUI is a
**normal `models.yaml` backend** (`backend: comfyui`, `engine: comfyui-server`).
That is the whole design decision, and everything else follows from it for free:

- **On-demand lifecycle (#422).** `startup: on_demand` + `idle_unload_minutes:
  15`. A backend used a few times a day shouldn't hold GPU memory against
  tower's always-on agentic/voice rotation. The first request spawns ComfyUI and
  waits for readiness; 15 idle minutes hand it back.
- **VRAM accounting.** `est_vram_mb: 5700` feeds the #375 overcommit warning —
  measured, not estimated (see below).
- **Admin + tray control**, per-backend logs at
  `data/logs/backend-flux1_local.log`, and inheritance across a hub restart —
  all inherited from `backend_process`, none of it written for this feature.

`src/comfyui_client.py` speaks ComfyUI's prompt API: `POST /prompt` with a
workflow graph, poll `GET /history/<id>`, then `GET /view` for the bytes.

Two details in that graph are load-bearing and fail *silently* if wrong:

- **`EmptySD3LatentImage`, not `EmptyLatentImage`.** FLUX uses a 16-channel
  latent; the SD-era node emits 4 and the sampler produces noise.
- **`cfg: 1.0` plus a `FluxGuidance` node.** FLUX [dev] is guidance-*distilled*.
  Real CFG is a no-op that merely doubles the work; the guidance scale (3.5)
  rides on the conditioning instead. The negative prompt is wired only because
  `KSampler` requires the input.

Output uses `PreviewImage` (ComfyUI's `temp` type) rather than `SaveImage`: the
hub hands the bytes back and has no reason to keep a copy, and `temp` is cleared
on ComfyUI startup — so the on-demand load/unload cycle garbage-collects
generations instead of growing an output directory nobody prunes.

ComfyUI is bound to **127.0.0.1 only**, unlike the llama/whisper backends'
`0.0.0.0`. It serves an unauthenticated, filesystem-capable web UI; the hub is
its only client, and other hosts reach image generation through this hub's API,
never ComfyUI's port.

### Install

```powershell
python scripts/install_comfyui.py                    # clone + isolated venv + torch cu130
python scripts/download_models.py --only flux1_local  # 17.25 GB checkpoint
```

The installer gives ComfyUI **its own venv** at `vendor/comfyui/.venv`. ComfyUI
pins a wide dependency surface that overlaps the hub's (transformers, numpy,
Pillow, pydantic…); resolving those into the hub's `.venv` would let an
image-engine bump silently downgrade a package the routing core depends on. A
second torch costs disk and buys a blast radius of zero.

Weights land in `models/comfyui/` (the shared, gitignored `models/` tree), not
under `vendor/` — so `install_comfyui.py --force` can wipe and rebuild the engine
without re-downloading 17 GB. ComfyUI finds them through the
`extra_model_paths.yaml` the installer generates.

**CUDA:** the RTX 5060 Ti is Blackwell (**sm_120**) and needs a CUDA ≥ 12.8
build; wheels come from PyTorch's `cu130` index. A default `pip install torch`
often resolves an older CUDA build with no sm_120 kernels and falls back to CPU
*silently* — ~20x slower, and only discovered at generation time. The installer's
`verify_cuda()` fails loudly on that instead, and can be re-run on its own with
`python scripts/install_comfyui.py --verify`.

### Sizes (#497)

`size` accepts a preset name or an explicit `"WIDTHxHEIGHT"` string, and the
presets live in one place — `src/image_sizes.py` — which is also what the
Playground dropdown renders, so the UI can't offer a size the API rejects.

| Preset | Pixels | Ratio | |
|---|---|---|---|
| `square` | 1024×1024 | 1:1 | native |
| `portrait` | 832×1216 | 2:3 | native |
| `landscape` | 1216×832 | 3:2 | native |
| `widescreen` | 1344×768 | 16:9 | native |
| `tall` | 768×1344 | 9:16 | native |
| `ultrawide` | 1536×640 | 21:9 | native |
| `square_hd` | 1440×1440 | 1:1 | native |
| `hd` | 1920×1088 | 16:9 | native |
| `square_2k` | 2048×2048 | 1:1 | upscaled |
| `4k` | 3840×2160 | 16:9 | upscaled |

Two rules explain most of the behaviour:

**Dimensions must be multiples of 16.** FLUX's VAE downsamples by 8 and the
transformer patchifies by a further 2; off-grid dimensions get padded and come
back with smeared edges. An off-grid request is **rejected with the nearest
valid pair**, not silently snapped — returning something other than what was
asked for, with no warning, is worse than a clear error. The visible
consequence: *there is no 1920×1080*, because 1080 isn't a multiple of 16. The
16:9 HD size is **1920×1088**, and asking for 1080 says so.

**Above ~2 MP the image is upscaled, not sampled natively.** FLUX.1 [dev] is
trained around 1 MP. Sampling natively at 4K (8.3 MP) is a *quality* cliff
before it is a speed one — attention cost grows with the square of the token
count and composition degrades into duplicated subjects. So a `4k` request is
sampled at the largest native-safe size of the same aspect ratio (1920×1088),
then upscaled with `4x-UltraSharp` and scaled to exactly 3840×2160. Preserving
the *aspect ratio* is what carries the composition through; the scale factor is
free. This is entirely internal — callers ask for `4k` and get 4K back.

`refine: true` adds a second low-denoise (0.25) pass over the upscaled image, so
it gains real detail rather than interpolated pixels. It is meaningful only for
upscaled sizes and is ignored for native ones. It is **expensive** — see below.

### Measured cost of each size

Measured on tower (RTX 5060 Ti, FLUX.1 [dev] fp8, 20 steps, warm backend,
quiet box). These are wall-clock through `POST /v1/images/generations`:

| Request | Sampled at | Time | Output |
|---|---|--:|--:|
| `square` 1024×1024 | 1024×1024 | **42 s** | 2.0 MB |
| `hd` 1920×1088 | 1920×1088 | **82 s** | 3.5 MB |
| `4k` 3840×2160 | 1920×1088 + upscale | **96 s** | 13.9 MB |
| `4k` + `refine` | 1920×1088 + upscale + 10-step refine | **384 s** | 13.3 MB |

So refining a 4K image costs about **4x** the un-refined one — the refine pass
samples at the full 8.3 MP, which is exactly the expensive thing the upscale
path exists to avoid doing for the *base* image. It buys real detail: an A/B at
a fixed seed (identical base image, differing only by the refine pass) measured
a mean absolute channel difference of 5.3/255 with peaks of 219/255, and the
map lettering in the test image goes from smudged to legible.

Two caveats worth knowing before turning it on:

- It is the only path here that can produce **visible artifacts**. In one of the
  samples taken during development, a large flat sky gradient came back with a
  faint vertical band — the high-resolution sampling cliff reappearing inside
  the refine pass. It did not recur on a detailed subject. Treat refine as
  "usually better, occasionally worse", not a free upgrade.
- **Beware benchmark caching.** ComfyUI caches node outputs, so timing two runs
  that share prompt + seed + sampled size measures only the second one's
  *upscale*. The first pass of these numbers reported 4K at 16 s for exactly
  that reason. Vary the seed when measuring.

`size` is accepted and **ignored** by `gemini_image`, deliberately: Imagen picks
its own dimensions, and 400-ing on a field every OpenAI client sends would break
existing callers. It is ignored *completely* — including values FLUX would
reject, such as `1920x1080`. Validating a field the backend then ignores would
mean rejecting an Imagen request with a message about FLUX's 16-pixel grid, a
rule that does not apply to the model being called. The Playground disables the
size control and says so when an Imagen model is selected rather than presenting
a control that does nothing.

### Limitations of the local path

- **No editing.** `POST /v1/images/edits` is gemini-only; img2img/inpainting
  needs a different workflow graph and was out of #492's scope. Asking for an
  edit with `flux1_local` returns a 400 that says exactly that, rather than the
  misleading "not an image-generation model".
- **It is not fast, warm or cold.** Measured on tower at 1024x1024 / 20 steps:
  **~55 s** for the first request after an idle unload (ComfyUI takes ~40 s to
  boot and accept work) and **~40 s** warm. Warm is not much better than cold
  because ComfyUI runs a *RAM-pressure cache*: it parks weights in system RAM
  between runs and re-streams them to the GPU each time, so the upload is paid
  per generation rather than once per load. That is also why the GPU footprint
  is far smaller than the checkpoint — see below.

### Measured footprint

Numbers below are sampled, not estimated — worth stating because the
"17.25 GB checkpoint on a 16 GB card" arithmetic suggests something much worse
than what actually happens.

Total board usage sampled once a second through a full generation, differenced
against a 7014 MB baseline with the row stopped (tower, 2026-08-11):

| | MB attributable to ComfyUI |
|---|---|
| idle, loaded, between runs | ~4000 |
| steady through sampling | ~4700 |
| peak | 5741 |

So it coexists comfortably with the rest of tower's rotation while running, and
`on_demand` is about not holding ~5 GB (plus its RSS) for a backend used a few
times a day — not about it being unable to fit.

## FLUX.2: dev 32B and klein 4B (#498)

FLUX.2 is **not** FLUX.1 with different weights. Three things differ, and each
one fails in a way that is hard to diagnose from a config file:

**It is a split-loader model.** FLUX.1 [dev] fp8 is one all-in-one checkpoint
(transformer + text encoders + VAE). FLUX.2 ships three separate files, so the
row carries `model_path` (the transformer) plus `extra_weights` for the text
encoder and VAE, and the graph uses `UnetLoaderGGUF`/`UNETLoader` +
`CLIPLoader(type="flux2")` + `VAELoader` instead of `CheckpointLoaderSimple`.

**It needs FLUX.2-specific nodes.** `EmptyFlux2LatentImage`, a
*resolution-aware* `Flux2Scheduler`, and `SamplerCustomAdvanced` + `BasicGuider`
— not `EmptySD3LatentImage`/`KSampler`. These names were read off a live
`/object_info`, not assumed; guessing here yields a runtime "value not in list"
at best and plausible noise at worst.

**dev and klein use different text encoders.** dev pairs with
Mistral-Small-3.2-24B, klein with **Qwen3-4B**. They are not interchangeable,
and swapping them does not fail at load — it fails deep in sampling with
`mat1 and mat2 shapes cannot be multiplied (512x15360 and 7680x3072)`.

### The 32B is slow, and that is the point

Measured on tower, 1024×1024, in isolation:

| | Cold | Warm | GPU peak |
|---|--:|--:|--:|
| `flux2_klein` (4B fp8) | ~100 s | **~25 s** | ~9.4 GB |
| `flux2_local` (32B Q4_K_M) | — | **~600 s** | ~13.7 GB |

The 32B streams roughly 30 GB of weights (18.7 GB transformer + 11.4 GB
encoder) from system RAM through a 16 GB card on every run. Its client timeout
is raised to 2400 s accordingly — the first measurement came in at **600.5 s**,
half a second past FLUX.1's 600 s budget, so sharing one timeout would have
failed the very first real request.

**Do not run the 32B at 4K.** The upscale path samples at ~2 MP, which on this
model means roughly 20 minutes before the upscale even starts. It is not
blocked, but nothing about it is pleasant.

### One ComfyUI process per row

Each image row gets its own port (`:8188` FLUX.1, `:8189` FLUX.2 dev, `:8190`
klein) and therefore its own ComfyUI process. One ComfyUI *could* serve every
model — the workflow names the weights per request — but this repo's process
layer keys start/stop, restart-inheritance, the idle watchdog and the port check
by model id, so a shared port would let one row's idle unload kill a server
another row is mid-request on.

That in turn requires per-process state: each instance is given its own
`--user-directory` and `--temp-directory` under `data/comfyui/<model_id>/`.
Sharing them means the second instance loses the SQLite race and starts degraded
with *"Could not acquire lock on database … Another ComfyUI process may already
be using it"*.

### Install

`scripts/install_comfyui.py` also provisions **city96's ComfyUI-GGUF** custom
node, pinned to an explicit commit (that repo publishes neither tags nor
releases, so `main` alone would not be reproducible). Stock ComfyUI cannot load
a `.gguf` diffusion model at all, so without it `flux2_local` is simply
unloadable.

It additionally **pins `transformers==4.57.6`** into ComfyUI's venv. ComfyUI asks
for `transformers>=4.50.3` with no upper bound, so pip installs 5.x, which
changed `MistralConverter.__init__` to take a positional `vocab_file` while
ComfyUI v0.31.0 still calls it as `MistralConverter(vocab=…)`. The mismatch is
invisible until a FLUX.2 generation actually loads the Mistral encoder, where it
surfaces as `CLIPLoader raised TypeError: MistralConverter.__init__() missing 1
required positional argument: 'vocab_file'`. Re-check this pin on a ComfyUI bump.

## The contract

Request (`POST /v1/images/generations`):

```json
{ "model": "gemini_image", "prompt": "a red apple on white",
  "n": 1, "response_format": "b64_json" }
```

Swap `"model"` for `"flux1_local"` (or `"flux"`) to run it locally — the request
and response shapes are identical.

- Only `n=1` and `response_format="b64_json"` are supported (400 otherwise).
- **No `size`.** Imagen controls the dimensions and ignores pixel-size hints —
  a `size` field on the request is silently dropped. Steer aspect ratio from
  the prompt text instead (e.g. "16:9"), which works.
- Any non-`gemini_image` model 400s — every other backend is text/audio only.

Response (OpenAI Images shape):

```json
{ "created": 1781500241, "data": [ { "b64_json": "<base64 image>" } ] }
```

Both routes land in the observability ring (`/v1/images/generations` and
`/v1/images/edits` are in `OBSERVABLE_PATHS`) exactly like `/v1/messages` and
the audio proxies.

## Playground

The admin SPA Playground has an **🖼️ Image generation** card: pick the model,
type a prompt, and Generate (then Download the result). Attaching a **reference
image** switches it to edit mode (with the slow/experimental warning above).
There is deliberately no size control — it had no effect. The card
proxies through `/admin/api/playground/generate_image` → the hub's own
`/v1/images/{generations,edits}` over loopback, so Playground image calls are
recorded in the live request ring like any external call. The card is hidden
on hosts with no image backend.

## Limitations / out of scope

- Image generation on the Claude or local (`qwen` / `gemma`) backends — no
  image-gen model exists there.
- Editing is procedural-agentic, not generative inpainting (see above).
- `gemini_image` requires the Gemini backend, i.e. the Windows host with `agy`
  signed in (same constraint as the `gemini_*` text rows).
