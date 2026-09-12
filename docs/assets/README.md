# Brand assets

Apache-2.0, same as the rest of the repository. Please feel free to use these to
refer to avgen: in a talk, a blog post, a comparison table, or a list of tools
you build on. Kindly do not use them to suggest that avgen endorses your
project, and do not use them as your own project's mark.

## What is here

| File | Use |
|---|---|
| `logo-light.svg` | The mark, on a light background |
| `logo-dark.svg` | The mark, on a dark background |
| `logo-light.png`, `logo-dark.png` | 512×512 raster, for contexts that cannot render SVG |
| `wordmark-light.svg`, `wordmark-dark.svg` | Mark plus the name |
| `wordmark-light.png`, `wordmark-dark.png` | 640×144 raster. **Prefer these when embedding**, see below |
| `favicon.svg` | Browser tab icon |
| `social-preview.png` | 1280×640, the repository's social card |

## Two files, not one recolouring file

The mark uses a stroke-and-knockout design. The front card is separated from the
frames behind it by a gap painted in the *background* colour. A single file using
`currentColor` cannot manage this, since `currentColor` gives you the ink colour
whereas the knockout needs to know what it is sitting on. Hence two files, with
the page choosing the right one:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo-dark.svg">
  <img src="logo-light.svg" alt="avgen" width="96" height="96">
</picture>
```

## The wordmark SVGs depend on a font

`wordmark-*.svg` sets the name in `<text>` with a font stack, so it renders with
whatever the viewer has installed and the letterforms shift between machines.
That is acceptable for a source file but wrong for anything published, so please
**embed the PNG instead**. It has no font dependency and looks the same
everywhere.

The mark on its own (`logo-*.svg`) contains no text and is safe to embed as SVG
anywhere.

## Colours

| Role | Light | Dark |
|---|---|---|
| Ink | `#14161a` | `#f2f3f5` |
| Ground / knockout | `#ffffff` | `#0d1117` |

You may reproduce the mark in a single flat colour when you need to: one ink on
one ground. Please do not recolour the two halves separately, add a gradient,
rotate it, or stretch it into a non-square box.

## What the mark means

Two outlined frames behind, one solid card in front, and inside that card a
short sequence of bars that step down.

The frames represent a clip, which is a stack of frames in time. The card in
front holds what they turn into, namely one token sequence. This is avgen's most
consequential design decision and the one the rest of the framework follows
from: the interior is sequence-first, and a dense five-dimensional grid exists
only at the boundary where the data and the codecs live. The mark is that
collapse, drawn out.

## Regenerating the raster exports

The PNGs are rendered from the SVGs using headless Chrome. This is worth knowing
because a rasteriser that silently drops `stroke` on `fill="none"` will give you
a plausible-looking but quite wrong image, and ImageMagick's built-in renderer
does exactly that.

```bash
python - <<'PY'
import subprocess, pathlib
def render(svg, out, w, h, bg):
    html = pathlib.Path("/tmp/_render.html")
    html.write_text(
        f'<html><body style="margin:0;background:{bg}">'
        f'<img src="file://{pathlib.Path(svg).resolve()}" '
        f'style="width:{w}px;height:{h}px;display:block"></body></html>'
    )
    subprocess.run([
        "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", f"--screenshot={out}",
        f"--window-size={w},{h}", str(html),
    ], check=True, capture_output=True)

render("logo-light.svg", "logo-light.png", 512, 512, "#ffffff")
render("logo-dark.svg", "logo-dark.png", 512, 512, "#0d1117")
render("wordmark-light.svg", "wordmark-light.png", 640, 144, "#ffffff")
render("wordmark-dark.svg", "wordmark-dark.png", 640, 144, "#0d1117")
PY
```
