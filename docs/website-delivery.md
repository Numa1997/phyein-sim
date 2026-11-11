# Delivering a finished simulation to the website

The rule: **the solver runs on GitHub (or Kaggle) and never in the visitor's browser.** A page shows three
files made from one finished run: a **film**, a **still**, and **one small JSON file**. This follows the
conventions of the SiteTest repository (phyeinusul.com); read its `CLAUDE.md` and `AGENTS.md` first, they win over
this note wherever they differ.

## 1. The three files

`tools/web_export.py <run_dir> <out_dir> --name <id>` writes, from a run that finished its own check:

| file | what it is | how big |
|---|---|---|
| `<id>.mp4` | the film, H.264, yuv420p, faststart, made by `tools/film.sh` | a few MB; the exporter refuses over `--max-mb` (default 20) |
| `<id>.png` | the still (last film frame) | under 1 MB |
| `<id>.json` | parameters, measured error and verdict, benchmark and simulated curves, and where the run came from (image, commit, date) | tens of kB |

Why these three: the film is the thing to look at, the still is the poster and the catalogue thumbnail, and the JSON
is the only place a number on the page may come from. The page draws its plot from the JSON. The same rule as SiteTest's
data panel holds: **the simulation writes each number where it computes it; the page only reads.** Nothing is recomputed for display.

## 2. Size limits (hard facts from SiteTest `docs/usul.md`)

- Cloudflare serves no static file above **25 MiB**; one such file makes the whole deploy fail and the live site keeps its old version.
- Before any merge with media: `find public -size +24M` must print nothing.
- Keep each film under about 10 MB: `film.sh` is H.264 with a constant-quality setting; lower the frame size or the frame rate rather than the quality if a film is too big.

## 3. Where the files go in SiteTest

- Film, JSON and still next to the page that uses them, in its own folder under `public/simulations/<id>/` (the repository's existing pattern, for example `public/simulations/fig13/video/`).
- The catalogue thumbnail is `public/images/sims/<id>.webp`, 800 × 500 px. Make it from the still (`cwebp` or Pillow).
- A new simulation page also needs one entry in `scripts/catalog.js` (`SIMULATIONS`, and its field in `SIM_FIELDS` if new), its number in the hand-written Fig. links at the top of every simulation page, and a docs page under `docs/simulations/`. The change is not finished until the docs say what the code now does.
- Every new page opens its head with the charset line, the Content-Security-Policy meta tag and the referrer meta tag, copied from an existing page. A film is `<video>` from the site's own origin, so the policy needs no new origin; check `media-src` / `default-src` in the copied tag before using it.
- Fonts, colours and layout are the house design (`docs/styles.md`); titles in Instrument Serif at weight 400, text in IBM Plex Sans, labels in IBM Plex Mono. No other font.
- Explanations are lecture notes set like a LaTeX document: everyday experience first, then the model, then the mathematics, with pictures. No question-and-answer format.
- German and Arabic pages are made by `tools/i18n/build.py`; follow `docs/i18n.md`.
- Run `python3 tools/inventory.py` as SiteTest's own notes say, so the file inventory stays true.

## 4. The proof the page must show

SiteTest asks for a three-way proof on any data table. For a simulation page that means three numbers, all visible:

1. the **stored value** (from the JSON, for example the measured error of the Strouhal number),
2. the **same quantity recomputed** from the stored inputs by the formula the page prints,
3. the **pixel** read back from the drawn plot, with the alpha channel checked first (a transparent pixel passes a "dark" test).

Aim for "identical"; name any rounding or float32 gap as exactly that. Where the drawing and the store differ (colour steps, smoothing, trails), say so on the page in one paragraph the reader can open.
Nothing on a running figure may react to the pointer merely crossing it: reading a point is a tool the reader switches on (click to pin, values in a fixed place outside the figure).

## 5. The flow, in order

1. A `run-case` workflow finishes on GitHub and uploads the artifact (case folder, `result.json`, `curves.json`, `film.mp4`).
2. Download the artifact; run `tools/web_export.py` on it inside the image (`PHYEIN_IMAGE` and `PHYEIN_COMMIT` set, so provenance is true).
3. In a branch of SiteTest, place the three files, add the page and the catalogue entry, build with `npm run build`, test in a real browser.
4. `find public -size +24M` prints nothing.
5. Merge to `main`. Cloudflare Workers Builds deploys by itself; verify the live site.

Steps 3 to 5 change the public site. They need the owner's explicit yes each time, except where SiteTest's `CLAUDE.md` already grants it (a finished, browser-tested change may be merged and published, since 3 November 2026).

## 6. Sentence for a future session

"Run the case `<name>` with the toolchain's run-case workflow, then export it with `tools/web_export.py` and add the page to SiteTest as `docs/website-delivery.md` describes; the page draws from the JSON and never solves anything in the browser."
