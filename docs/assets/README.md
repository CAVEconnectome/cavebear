# cavebear brand assets

The bear is built only from flat triangles that meet vertex to vertex (no T-junctions).
The SVGs keep that geometry, with one `<path>` per colour so same-colour neighbours show no anti-aliasing seams.

## Files

| File | Use |
| --- | --- |
| `cavebear-lockup.svg` | Logo + wordmark (README header, docs site). Works on light and dark backgrounds. Text is outlined, so no font is needed. |
| `cavebear-lockup.png`, `cavebear-lockup@2x.png` | Raster lockup, 564×110 and 1129×220. |
| `cavebear-logo.svg`, `cavebear-logo.png` | Bear only, native 405×329 proportions. |
| `cavebear-icon.svg` | Bear centred on a transparent 512×512 square. |
| `cavebear-icon-{16,32,48,64,128,192,256,512,1024}.png` | Raster icons (transparent). |
| `cavebear-icon-tile.svg`, `cavebear-icon-tile-512.png` | Bear on a cream square. For dark UIs, app/org avatars and social cards. |
| `apple-touch-icon.png` | 180×180 cream-tile icon. |
| `favicon.ico` | 16/32/48 multi-size favicon. |

## Colours

| | Hex |
| --- | --- |
| Wordmark "cave" (amber) | `#b67b30` |
| Wordmark "bear" (burnt orange) | `#e1640d` |
| Bear palette | `#2c1910` `#4c2c1d` `#714429` `#b67b30` `#e9a754` `#f38f26` `#e1640d` |
| Tile background (cream) | `#f1e9dc` |

Wordmark: Space Grotesk Bold, tracking −0.04em, cap/ascender sized at 100 px against a 110 px bear, 10 px gap.

## Snippets

README header:

```html
<p align="center"><img src="docs/assets/cavebear-lockup.svg" alt="cavebear" height="88"></p>
```

Favicons:

```html
<link rel="icon" href="assets/favicon.ico" sizes="any">
<link rel="icon" href="assets/cavebear-icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
```

Below about 32 px the bear's detail breaks down; prefer the tile version at small sizes on dark backgrounds.
