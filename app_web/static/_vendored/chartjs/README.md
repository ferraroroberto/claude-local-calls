# `chartjs` — trend charts (Cld tab, telemetry sparklines)

Vendored **byte-for-byte** from the upstream
[Chart.js](https://www.chartjs.org/) v4.4.7 UMD build (MIT license),
downloaded from jsdelivr's npm mirror and verified via `sha256sum` at copy
time (local-llm-hub#451): previously the SPA's only third-party dependency
loaded from a CDN (`https://cdn.jsdelivr.net/npm/chart.js@4.4.7/…`, no
`integrity`/`crossorigin` pin) while every other component here — `xterm`,
`icons`, `nav`, `switch`, `modal`, `disclosure`, `empty-state`, `button`,
`card` — already lived under `_vendored/`. That made the admin UI's
chart-bearing tabs silently degrade on any machine without internet,
including the headless fleet satellites this console exists to administer.

## Files

| File | Role |
| --- | --- |
| `chart.umd.min.js` | The full Chart.js bundle (core + every built-in chart type/scale/plugin) — exposed as `window.Chart` when loaded via a plain `<script>` tag. |

SHA-256 of `chart.umd.min.js`:
`206b6e8bb00fc7bba2c7ee80ca41db3e9e05ba7be0aa35abeba9cfd5357f5d0e`

## Usage

```html
<script src="/admin/static/_vendored/chartjs/chart.umd.min.js"></script>
```

```js
const chart = new window.Chart(ctx, { type: "line", data, options });
```

The file carries an upstream `//# sourceMappingURL=chart.umd.js.map` comment;
the `.map` itself is not vendored (browsers only fetch it when DevTools'
source panel is opened, and a missing map 404s harmlessly — same as upstream
distributions that ship the minified file without its map).

## Don't diverge

Do not hand-edit `chart.umd.min.js` — to pick up a newer Chart.js release,
re-download the pinned version's `dist/chart.umd.min.js` from npm/jsdelivr,
re-verify with `sha256sum`, and update the hash above.
