# Image generation (Google Imagen via the Antigravity CLI)

The hub can generate images through `POST /v1/images/generations` (OpenAI
Images shape). This note records *what backend actually does the work* and
*why the contract looks the way it does* — both differ from the obvious
first guess.

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

## The contract

Request (`POST /v1/images/generations`):

```json
{ "model": "gemini_image", "prompt": "a red apple on white",
  "n": 1, "response_format": "b64_json" }
```

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
