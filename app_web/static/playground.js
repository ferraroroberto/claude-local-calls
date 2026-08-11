/* Playground tab — stacked label/input pairs, More-options details,
 * segmented max-tokens with a numeric override, full-width Send.
 */

import { els, state } from './state.js';
import { api, jsonApi, toast } from './api.js';
import { setSwitch } from './_vendored/switch/switch.js';

let ttsModels = [];
let lastTtsSample = '';

export async function fetchPlaygroundModels() {
  try {
    const body = await jsonApi('/admin/api/playground/models');
    const models = body.models || [];
    if (!els.playgroundModel) return;
    els.playgroundModel.innerHTML = '';
    models.forEach(function (m) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.display_name + ' (' + m.backend + ')';
      opt.dataset.backend = m.backend;
      els.playgroundModel.appendChild(opt);
    });
  } catch (_) { /* ignore */ }
}

let imageModels = [];
let imageSizes = [];

export async function fetchImageModels() {
  try {
    const body = await jsonApi('/admin/api/playground/image_models');
    const models = body.models || [];
    imageModels = models;
    imageSizes = body.sizes || [];
    if (!els.imageModel) return;
    // No image-generation backend on this host → hide the whole card.
    if (els.imageCard) els.imageCard.hidden = models.length === 0;
    els.imageModel.innerHTML = '';
    models.forEach(function (m) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.display_name + ' — ' + m.where;
      els.imageModel.appendChild(opt);
    });
    // The API returns size-capable (local) models first, so the first entry is
    // already the local-by-default choice; select it explicitly rather than
    // relying on the browser's implicit first-option behaviour.
    if (models.length) els.imageModel.value = models[0].id;

    if (els.imageSize) {
      els.imageSize.innerHTML = '';
      imageSizes.forEach(function (s) {
        const opt = document.createElement('option');
        opt.value = s.name;
        opt.textContent = s.label + ' · ' + s.width + '×' + s.height +
          ' (' + s.ratio + ')' + (s.upscaled ? ' — upscaled' : '');
        opt.dataset.upscaled = s.upscaled ? '1' : '';
        els.imageSize.appendChild(opt);
      });
      if (body.default_size) {
        const match = imageSizes.find(function (s) {
          return s.name === body.default_size ||
            (s.width + 'x' + s.height) === body.default_size;
        });
        if (match) els.imageSize.value = match.name;
      }
    }
    _syncImageCapabilities();
  } catch (_) { /* ignore */ }
}

function _selectedImageModel() {
  if (!els.imageModel) return null;
  return imageModels.find(function (m) { return m.id === els.imageModel.value; }) || null;
}

function _selectedImageSize() {
  if (!els.imageSize) return null;
  return imageSizes.find(function (s) { return s.name === els.imageSize.value; }) || null;
}

// Keep the controls honest about what the chosen model can actually do: a
// backend that ignores size must not present a size control that looks live,
// and refine only means anything on an upscaled size.
function _syncImageCapabilities() {
  const model = _selectedImageModel();
  const size = _selectedImageSize();
  const supportsSize = !!(model && model.supports_size);

  if (els.imageModelNote) els.imageModelNote.textContent = model ? model.note : '';

  if (els.imageSize) els.imageSize.disabled = !supportsSize;
  if (els.imageSizeNote) {
    if (!supportsSize) {
      els.imageSizeNote.textContent =
        'This model picks its own dimensions — ask for an aspect ratio in the prompt instead (e.g. “16:9”).';
    } else if (size && size.upscaled) {
      els.imageSizeNote.textContent =
        'Above the model’s native resolution: generated at a native size, then upscaled to ' +
        size.width + '×' + size.height + '. Slower.';
    } else if (size) {
      els.imageSizeNote.textContent = size.megapixels + ' MP, generated natively.';
    } else {
      els.imageSizeNote.textContent = '';
    }
  }

  const canRefine = supportsSize && !!(size && size.upscaled);
  if (els.imageRefineRow) els.imageRefineRow.hidden = !canRefine;

  if (els.imageEditNote) {
    els.imageEditNote.textContent = (model && model.supports_edit)
      ? 'Editing is supported on this model, but is procedural and slow — minutes, not seconds.'
      : 'Editing is not supported on this model — attach a reference image only with an Imagen model.';
  }
}

export async function fetchTtsModels() {
  if (els.ttsCard) els.ttsCard.dataset.state = 'loading';
  try {
    const body = await jsonApi('/admin/api/playground/tts_models');
    const models = body.models || [];
    ttsModels = models;
    if (!els.ttsModel) return;
    els.ttsModel.innerHTML = '';
    // No TTS backend configured on this host → hide the whole card.
    if (els.ttsCard) els.ttsCard.hidden = models.length === 0;
    models.forEach(function (m) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.display_name + (m.engine ? ' (' + m.engine + ')' : '') +
        (m.reachable ? '' : ' — stopped');
      opt.dataset.engine = m.engine || '';
      opt.disabled = !m.reachable;
      els.ttsModel.appendChild(opt);
    });
    const firstReachable = models.find(function (m) { return m.reachable; });
    if (firstReachable) els.ttsModel.value = firstReachable.id;
    els.ttsModel.disabled = !firstReachable;
    if (els.ttsCard) els.ttsCard.dataset.state = models.length ? 'ready' : 'empty';
    _syncTtsCapabilities(true);
  } catch (_) {
    if (els.ttsCard) els.ttsCard.dataset.state = 'error';
    if (els.ttsAvailability) els.ttsAvailability.textContent = 'Voice model status could not be loaded.';
  }
}

/* Hidden-input file picker (#215): a ghost "Choose file" button proxies the
 * native input; the selected filename shows in the label to its right. */
function wireFilePicker(input, btn, nameEl) {
  if (!input || !btn) return;
  btn.addEventListener('click', function () { input.click(); });
  input.addEventListener('change', function () {
    if (!nameEl) return;
    const f = input.files && input.files[0];
    nameEl.textContent = f ? f.name : 'No file selected';
  });
}

export function wirePlayground() {
  if (els.playgroundSendBtn) {
    els.playgroundSendBtn.addEventListener('click', sendPrompt);
  }
  wireFilePicker(els.playgroundAttachment, els.playgroundAttachmentBtn, els.playgroundAttachmentName);
  wireFilePicker(els.imageAttachment, els.imageAttachmentBtn, els.imageAttachmentName);
  wireTts();
  wireImage();
  if (els.playgroundClearBtn) {
    els.playgroundClearBtn.addEventListener('click', function () {
      els.playgroundReply.textContent = '';
      els.playgroundUsage.innerHTML = '';
      els.playgroundLatency.textContent = '';
    });
  }
  // Segmented max-tokens — clicking a preset highlights it AND mirrors
  // the value into the numeric override. Typing into the override clears
  // the active preset highlight (numeric override wins).
  if (els.playgroundMaxTokensSeg) {
    els.playgroundMaxTokensSeg.addEventListener('click', function (ev) {
      const btn = ev.target.closest('button[data-value]');
      if (!btn) return;
      const val = parseInt(btn.dataset.value, 10) || 512;
      els.playgroundMaxTokensSeg.querySelectorAll('button').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
      if (els.playgroundMaxTokens) els.playgroundMaxTokens.value = String(val);
    });
  }
  if (els.playgroundMaxTokens) {
    els.playgroundMaxTokens.addEventListener('input', function () {
      // User typed into the override — clear preset highlights so it's
      // visually clear the value comes from the input, not a preset.
      if (!els.playgroundMaxTokensSeg) return;
      const current = String(parseInt(els.playgroundMaxTokens.value, 10) || 0);
      els.playgroundMaxTokensSeg.querySelectorAll('button').forEach(function (b) {
        b.classList.toggle('active', b.dataset.value === current);
      });
    });
  }
}

// The Stream control is the vendored fleet switch (_vendored/switch) —
// a .toggle button whose state lives in aria-checked, not a checkbox.
function _streamOn() {
  return !!(els.ttsStream && els.ttsStream.getAttribute('aria-checked') === 'true');
}

function _selectedTtsModel() {
  if (!els.ttsModel) return null;
  return ttsModels.find(function (model) { return model.id === els.ttsModel.value; }) || null;
}

function _setTtsSample(capabilities, language) {
  if (!els.ttsInput) return;
  const sample = (capabilities.sample_text || {})[language] || '';
  const current = els.ttsInput.value || '';
  if (!current.trim() || current === lastTtsSample) {
    els.ttsInput.value = sample;
    lastTtsSample = sample;
  }
}

function _populateTtsVoices(capabilities, preferredVoice) {
  if (!els.ttsVoice || !els.ttsLanguage) return;
  const language = els.ttsLanguage.value;
  const voices = (capabilities.voices || []).filter(function (voice) {
    return voice.language === language;
  });
  els.ttsVoice.innerHTML = '';
  voices.forEach(function (voice) {
    const opt = document.createElement('option');
    opt.value = voice.id;
    opt.textContent = voice.label + (voice.gender ? ' · ' + voice.gender : '');
    els.ttsVoice.appendChild(opt);
  });
  const wanted = voices.some(function (voice) { return voice.id === preferredVoice; })
    ? preferredVoice
    : voices[0] && voices[0].id;
  if (wanted) els.ttsVoice.value = wanted;
}

function _syncTtsControls(capabilities) {
  const controls = capabilities.controls || {};
  if (els.ttsStreamGroup) els.ttsStreamGroup.hidden = !controls.stream;
  if (els.ttsStream) {
    els.ttsStream.disabled = !controls.stream;
    if (!controls.stream) setSwitch(els.ttsStream, false);
  }
  if (els.ttsSpeedGroup) els.ttsSpeedGroup.hidden = !controls.speed;
  if (els.ttsExaggerationGroup) els.ttsExaggerationGroup.hidden = !controls.exaggeration;
  if (els.ttsCfgWeightGroup) els.ttsCfgWeightGroup.hidden = !controls.cfg_weight;
}

function _syncTtsCapabilities(modelChanged) {
  const model = _selectedTtsModel();
  const available = !!(model && model.reachable);
  if (els.ttsSpeakBtn) els.ttsSpeakBtn.disabled = !available;
  if (els.ttsAvailability) {
    els.ttsAvailability.textContent = available
      ? 'Ready.'
      : (ttsModels.length ? 'All configured voice models are stopped.' : 'No voice models configured.');
  }
  if (!model) {
    if (els.ttsLanguage) els.ttsLanguage.innerHTML = '';
    if (els.ttsVoice) els.ttsVoice.innerHTML = '';
    _syncTtsControls({});
    return;
  }

  const capabilities = model.capabilities || {};
  const previousLanguage = modelChanged ? '' : (els.ttsLanguage && els.ttsLanguage.value);
  if (els.ttsLanguage) {
    els.ttsLanguage.innerHTML = '';
    (capabilities.languages || []).forEach(function (language) {
      const opt = document.createElement('option');
      opt.value = language.id;
      opt.textContent = language.label;
      els.ttsLanguage.appendChild(opt);
    });
    const languageIds = (capabilities.languages || []).map(function (language) { return language.id; });
    els.ttsLanguage.value = languageIds.includes(previousLanguage)
      ? previousLanguage
      : (capabilities.default_language || languageIds[0] || '');
  }
  _populateTtsVoices(capabilities, modelChanged ? capabilities.default_voice : els.ttsVoice.value);
  _syncTtsControls(capabilities);
  _setTtsSample(capabilities, els.ttsLanguage ? els.ttsLanguage.value : '');
}

function wireTts() {
  // Live value readouts for the range sliders.
  if (els.ttsSpeed && els.ttsSpeedVal) {
    els.ttsSpeed.addEventListener('input', function () {
      els.ttsSpeedVal.textContent = Number(els.ttsSpeed.value).toFixed(2).replace(/0$/, '');
    });
  }
  if (els.ttsExaggeration && els.ttsExaggerationVal) {
    els.ttsExaggeration.addEventListener('input', function () {
      els.ttsExaggerationVal.textContent = els.ttsExaggeration.value;
    });
  }
  if (els.ttsCfgWeight && els.ttsCfgWeightVal) {
    els.ttsCfgWeight.addEventListener('input', function () {
      els.ttsCfgWeightVal.textContent = els.ttsCfgWeight.value;
    });
  }
  if (els.ttsModel) {
    els.ttsModel.addEventListener('change', function () { _syncTtsCapabilities(true); });
  }
  if (els.ttsLanguage) {
    els.ttsLanguage.addEventListener('change', function () {
      const model = _selectedTtsModel();
      if (!model) return;
      _populateTtsVoices(model.capabilities || {}, '');
      _setTtsSample(model.capabilities || {}, els.ttsLanguage.value);
    });
  }
  if (els.ttsStream) {
    els.ttsStream.addEventListener('click', function () {
      if (els.ttsStream.disabled) return;
      setSwitch(els.ttsStream, !_streamOn());
    });
  }
  if (els.ttsSpeakBtn) {
    els.ttsSpeakBtn.addEventListener('click', speak);
  }
}

function ttsFormData(text, modelId, streaming) {
  const fd = new FormData();
  fd.append('model', modelId);
  fd.append('input', text);
  fd.append('voice', (els.ttsVoice.value || '').trim());
  fd.append('response_format', streaming ? 'pcm' : (els.ttsFormat ? els.ttsFormat.value : 'wav'));
  if (streaming) fd.append('stream', 'true');
  const model = _selectedTtsModel();
  const controls = (model && model.capabilities && model.capabilities.controls) || {};
  if (controls.speed && els.ttsSpeed) fd.append('speed', els.ttsSpeed.value);
  if (controls.exaggeration && els.ttsExaggeration) fd.append('exaggeration', els.ttsExaggeration.value);
  if (controls.cfg_weight && els.ttsCfgWeight) fd.append('cfg_weight', els.ttsCfgWeight.value);
  return fd;
}

async function speak() {
  if (!els.ttsModel || !els.ttsModel.value) {
    toast('No TTS model available.', 'error');
    return;
  }
  const text = (els.ttsInput.value || '').trim();
  if (!text) {
    toast('Text is empty.', 'error');
    return;
  }
  els.ttsSpeakBtn.disabled = true;
  els.ttsLatency.textContent = 'synthesizing…';

  const streaming = _streamOn();
  if (streaming) {
    try {
      await speakStream(text);
    } catch (exc) {
      els.ttsLatency.textContent = '';
      toast(String(exc.message || exc), 'error');
    } finally {
      els.ttsSpeakBtn.disabled = false;
    }
    return;
  }

  const fd = ttsFormData(text, els.ttsModel.value, false);

  const t0 = performance.now();
  try {
    const res = await api('/admin/api/playground/speak', { method: 'POST', body: fd });
    if (!res.ok) {
      let msg = 'HTTP ' + res.status;
      try { const b = await res.json(); msg = b.detail || msg; } catch (_) { /* ignore */ }
      els.ttsLatency.textContent = '';
      toast(msg, 'error');
      return;
    }
    const blob = await res.blob();
    const elapsed = (performance.now() - t0).toFixed(0);
    els.ttsLatency.textContent = elapsed + ' ms · ' + Math.round(blob.size / 1024) + ' KB';
    if (els.ttsAudio.dataset.url) URL.revokeObjectURL(els.ttsAudio.dataset.url);
    const url = URL.createObjectURL(blob);
    els.ttsAudio.dataset.url = url;
    els.ttsAudio.src = url;
    els.ttsAudio.hidden = false;
    els.ttsAudio.play().catch(function () { /* autoplay may be blocked; controls remain */ });
  } catch (exc) {
    els.ttsLatency.textContent = '';
    toast(String(exc.message || exc), 'error');
  } finally {
    els.ttsSpeakBtn.disabled = false;
  }
}

// Streaming synthesis: request headerless PCM16 with stream_format=audio and
// schedule each chunk on a Web Audio timeline so playback starts as soon as
// the first frames arrive. Reports time-to-first-audio vs total — the whole
// point of the streaming endpoint (issue #102). The native <audio> element
// can't consume a chunked POST, hence Web Audio rather than ttsAudio.src.
async function speakStream(text) {
  const fd = ttsFormData(text, els.ttsModel.value, true);

  els.ttsAudio.hidden = true;            // streamed playback uses Web Audio
  // Create + resume the AudioContext *inside* the click gesture (before the
  // first await) so browser autoplay policy lets it produce sound.
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx();
  try { await ctx.resume(); } catch (_) { /* ignore */ }

  const t0 = performance.now();
  const res = await api('/admin/api/playground/speak', { method: 'POST', body: fd });
  if (!res.ok) {
    let msg = 'HTTP ' + res.status;
    try { const b = await res.json(); msg = b.detail || msg; } catch (_) { /* ignore */ }
    els.ttsLatency.textContent = '';
    ctx.close().catch(function () { /* ignore */ });
    toast(msg, 'error');
    return;
  }

  const sampleRate = parseInt(res.headers.get('X-Sample-Rate') || '24000', 10) || 24000;
  let playHead = ctx.currentTime + 0.08;  // small lead-in to avoid underrun
  let ttfa = null;
  let totalSamples = 0;
  let leftover = new Uint8Array(0);
  const reader = res.body.getReader();

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value || value.length === 0) continue;
    if (ttfa === null) {
      ttfa = performance.now() - t0;
      els.ttsLatency.textContent = 'first audio ' + ttfa.toFixed(0) + ' ms · playing…';
    }
    // Merge any odd trailing byte from the previous chunk, then split into
    // whole int16 samples (carry the remainder forward).
    const merged = new Uint8Array(leftover.length + value.length);
    merged.set(leftover, 0);
    merged.set(value, leftover.length);
    const usable = merged.length - (merged.length % 2);
    leftover = merged.slice(usable);
    if (usable === 0) continue;
    const i16 = new Int16Array(merged.buffer.slice(0, usable));
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const buf = ctx.createBuffer(1, f32.length, sampleRate);
    buf.copyToChannel(f32, 0);
    const node = ctx.createBufferSource();
    node.buffer = buf;
    node.connect(ctx.destination);
    node.start(playHead);
    playHead += buf.duration;
    totalSamples += f32.length;
  }

  const total = performance.now() - t0;
  const audioSec = totalSamples / sampleRate;
  els.ttsLatency.textContent =
    'first audio ' + (ttfa || 0).toFixed(0) + ' ms · total ' + total.toFixed(0) +
    ' ms · ' + audioSec.toFixed(1) + ' s audio';
  // Let scheduled buffers finish, then release the context.
  setTimeout(function () { ctx.close().catch(function () { /* ignore */ }); },
    Math.max(0, (playHead - ctx.currentTime) * 1000) + 500);
}

function wireImage() {
  if (els.imageGenBtn) els.imageGenBtn.addEventListener('click', generateImage);
  if (els.imageModel) els.imageModel.addEventListener('change', _syncImageCapabilities);
  if (els.imageSize) els.imageSize.addEventListener('change', _syncImageCapabilities);
  // Same vendored fleet switch as the TTS Stream control — state lives in
  // aria-checked, not a checkbox.
  if (els.imageRefine) {
    els.imageRefine.addEventListener('click', function () {
      const on = els.imageRefine.getAttribute('aria-checked') !== 'true';
      els.imageRefine.setAttribute('aria-checked', on ? 'true' : 'false');
      els.imageRefine.classList.toggle('on', on);
      const label = els.imageRefine.querySelector('.toggle-label');
      if (label) label.textContent = on ? 'ON' : 'OFF';
    });
  }
  if (els.imageClearBtn) {
    els.imageClearBtn.addEventListener('click', function () {
      if (els.imagePreview) {
        if (els.imagePreview.dataset.url) URL.revokeObjectURL(els.imagePreview.dataset.url);
        els.imagePreview.removeAttribute('src');
        els.imagePreview.hidden = true;
      }
      if (els.imageDownloadRow) els.imageDownloadRow.hidden = true;
      if (els.imageLatency) els.imageLatency.textContent = '';
    });
  }
}

async function generateImage() {
  if (!els.imageModel || !els.imageModel.value) {
    toast('No image model available.', 'error');
    return;
  }
  const prompt = (els.imagePrompt.value || '').trim();
  if (!prompt) {
    toast('Prompt is empty.', 'error');
    return;
  }
  const editing = !!(els.imageAttachment && els.imageAttachment.files && els.imageAttachment.files[0]);

  const model = _selectedImageModel();
  const size = _selectedImageSize();
  const refining = !!(els.imageRefine &&
    els.imageRefine.getAttribute('aria-checked') === 'true' &&
    size && size.upscaled);

  const fd = new FormData();
  fd.append('model', els.imageModel.value);
  fd.append('prompt', prompt);
  if (editing) fd.append('image', els.imageAttachment.files[0]);
  // Only send size to a backend that honours it — the hub accepts and ignores
  // it elsewhere, but sending it would imply a control that did nothing.
  if (!editing && size && model && model.supports_size) {
    fd.append('size', size.name);
    if (refining) fd.append('refine', 'true');
  }

  els.imageGenBtn.disabled = true;
  els.imageLatency.textContent = editing
    ? 'editing… (procedural — can take minutes)'
    : (size && size.upscaled)
      ? 'generating then upscaling to ' + size.width + '×' + size.height +
        (refining ? ' + refining' : '') + '… (slow)'
      : 'generating…';

  const t0 = performance.now();
  try {
    const res = await api('/admin/api/playground/generate_image', { method: 'POST', body: fd });
    if (!res.ok) {
      let msg = 'HTTP ' + res.status;
      try { const b = await res.json(); msg = b.detail || msg; } catch (_) { /* ignore */ }
      els.imageLatency.textContent = '';
      toast(msg, 'error');
      return;
    }
    const blob = await res.blob();
    const elapsed = (performance.now() - t0).toFixed(0);
    const seconds = (elapsed / 1000).toFixed(1);
    els.imageLatency.textContent = seconds + ' s · ' + Math.round(blob.size / 1024) + ' KB';
    if (els.imagePreview.dataset.url) URL.revokeObjectURL(els.imagePreview.dataset.url);
    const url = URL.createObjectURL(blob);
    els.imagePreview.dataset.url = url;
    els.imagePreview.src = url;
    els.imagePreview.hidden = false;
    // Report the dimensions actually delivered, read off the decoded image
    // rather than echoing what was requested — that way a snapped, upscaled or
    // backend-chosen size is visible instead of surprising. Works for both
    // backends, including Imagen, which never reports its own size.
    els.imagePreview.addEventListener('load', function onLoad() {
      els.imagePreview.removeEventListener('load', onLoad);
      const w = els.imagePreview.naturalWidth;
      const h = els.imagePreview.naturalHeight;
      if (w && h) {
        els.imageLatency.textContent =
          seconds + ' s · ' + w + '×' + h + ' · ' + Math.round(blob.size / 1024) + ' KB';
      }
    });
    if (els.imageDownload) {
      const ext = (blob.type && blob.type.indexOf('jpeg') >= 0) ? 'jpg'
        : (blob.type && blob.type.indexOf('webp') >= 0) ? 'webp' : 'png';
      els.imageDownload.href = url;
      els.imageDownload.download = 'generated-' + Date.now() + '.' + ext;
      if (els.imageDownloadRow) els.imageDownloadRow.hidden = false;
    }
  } catch (exc) {
    els.imageLatency.textContent = '';
    toast(String(exc.message || exc), 'error');
  } finally {
    els.imageGenBtn.disabled = false;
  }
}

async function sendPrompt() {
  const modelSel = els.playgroundModel;
  const prompt = (els.playgroundPrompt.value || '').trim();
  if (!modelSel.value) {
    toast('Pick a model first.', 'error');
    return;
  }
  if (!prompt) {
    toast('Prompt is empty.', 'error');
    return;
  }
  els.playgroundSendBtn.disabled = true;
  els.playgroundReply.textContent = '…';
  els.playgroundUsage.innerHTML = '';
  els.playgroundLatency.textContent = 'sending…';

  const fd = new FormData();
  fd.append('model', modelSel.value);
  fd.append('prompt', prompt);
  fd.append('max_tokens', String(parseInt(els.playgroundMaxTokens.value, 10) || 512));
  const system = (els.playgroundSystem.value || '').trim();
  if (system) fd.append('system', system);
  if (els.playgroundAttachment && els.playgroundAttachment.files && els.playgroundAttachment.files[0]) {
    fd.append('attachment', els.playgroundAttachment.files[0]);
  }

  const t0 = performance.now();
  try {
    const res = await api('/admin/api/playground/send', { method: 'POST', body: fd });
    const body = await res.json().catch(function () { return null; });
    if (!res.ok) {
      const msg = (body && body.detail) || ('HTTP ' + res.status);
      els.playgroundReply.textContent = '[' + res.status + '] ' + msg;
      els.playgroundLatency.textContent = '';
      toast(msg, 'error');
      return;
    }
    const elapsed = (performance.now() - t0).toFixed(0);
    els.playgroundLatency.textContent = elapsed + ' ms';
    els.playgroundReply.textContent = body.text || '(no text)';
    renderUsage(body.usage || {});
  } catch (exc) {
    els.playgroundReply.textContent = String(exc.message || exc);
    toast(String(exc.message || exc), 'error');
  } finally {
    els.playgroundSendBtn.disabled = false;
  }
}

function renderUsage(usage) {
  const grid = els.playgroundUsage;
  if (!grid) return;
  grid.innerHTML = '';
  const rows = [
    ['Input', usage.input_tokens || 0],
    ['Output', usage.output_tokens || 0],
    ['Cache read', usage.cache_read_input_tokens || 0],
    ['Cache write', usage.cache_creation_input_tokens || 0],
  ];
  rows.forEach(function (r) {
    const div = document.createElement('div');
    div.innerHTML = '<span class="muted">' + r[0] + '</span><span>' + r[1] + '</span>';
    grid.appendChild(div);
  });
}
