"""Build figure gallery from registry: thumbnails, HTML, and validation."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from dynachaos.pipelines.registry import SECTION_ORDER, SECTION_SPECS

# Import gallery_meta from same directory
sys.path.insert(0, str(Path(__file__).parent))
from gallery_meta import CAPTIONS, SECTION_TITLES


def main() -> int:
    """Build gallery: thumbnails, HTML, validate all figures exist and have captions."""
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    figures_dir = project_root / "figures"
    thumbs_dir = figures_dir / "thumbs"
    site_dir = project_root / "site"
    site_thumbs_dir = site_dir / "thumbs"
    site_full_dir = site_dir / "full"

    # Validation: check all registry PNGs exist and have captions
    missing_files = []
    missing_captions = []

    for section_id in SECTION_ORDER:
        spec = SECTION_SPECS[section_id]
        for output_file in spec.output_files:
            if not output_file.endswith(".png"):
                continue

            # Check file exists
            png_path = figures_dir / section_id / output_file
            if not png_path.exists():
                missing_files.append(str(png_path))

            # Check caption exists
            caption_key = f"{section_id}/{output_file}"
            if caption_key not in CAPTIONS:
                missing_captions.append(caption_key)

    if missing_files:
        print("ERROR: Missing PNG files:", file=sys.stderr)
        for fpath in missing_files:
            print(f"  {fpath}", file=sys.stderr)
        return 1

    if missing_captions:
        print("ERROR: Missing captions for:", file=sys.stderr)
        for key in missing_captions:
            print(f"  {key}", file=sys.stderr)
        return 1

    # Create directories
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    site_thumbs_dir.mkdir(parents=True, exist_ok=True)
    site_full_dir.mkdir(parents=True, exist_ok=True)

    # Generate thumbnails and collect file sizes
    thumb_count = 0
    thumb_bytes = 0
    force = "--force" in sys.argv

    for section_id in SECTION_ORDER:
        section_thumbs_dir = thumbs_dir / section_id
        section_thumbs_dir.mkdir(parents=True, exist_ok=True)

        spec = SECTION_SPECS[section_id]
        for output_file in spec.output_files:
            if not output_file.endswith(".png"):
                continue

            png_path = figures_dir / section_id / output_file
            webp_name = output_file.replace(".png", ".webp")
            webp_path = section_thumbs_dir / webp_name

            # Skip if webp exists and is newer (unless --force)
            if not force and webp_path.exists():
                if webp_path.stat().st_mtime >= png_path.stat().st_mtime:
                    thumb_bytes += webp_path.stat().st_size
                    thumb_count += 1
                    continue

            # Generate thumbnail
            img = Image.open(png_path)
            # Resize to max width 900px, preserve aspect ratio, never upscale
            max_width = 900
            if img.width > max_width:
                scale = max_width / img.width
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            img.save(webp_path, "WebP", quality=82, method=6)
            thumb_bytes += webp_path.stat().st_size
            thumb_count += 1

    # Copy full PNGs to site
    png_count = 0
    for section_id in SECTION_ORDER:
        section_full_dir = site_full_dir / section_id
        section_full_dir.mkdir(parents=True, exist_ok=True)

        spec = SECTION_SPECS[section_id]
        for output_file in spec.output_files:
            if not output_file.endswith(".png"):
                continue

            src = figures_dir / section_id / output_file
            dst = section_full_dir / output_file
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                dst.write_bytes(src.read_bytes())
            png_count += 1

    # Copy thumbnails to site
    for section_id in SECTION_ORDER:
        src_section = thumbs_dir / section_id
        dst_section = site_thumbs_dir / section_id
        dst_section.mkdir(parents=True, exist_ok=True)

        if src_section.exists():
            for webp_file in src_section.glob("*.webp"):
                dst = dst_section / webp_file.name
                if not dst.exists() or dst.stat().st_mtime < webp_file.stat().st_mtime:
                    dst.write_bytes(webp_file.read_bytes())

    # Build HTML
    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang='en'>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='UTF-8'>")
    html_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    html_parts.append("<title>dynachaos — reproduction gallery</title>")
    html_parts.append("<style>")

    # CSS
    html_parts.append("""
:root {
  --bg: #ffffff;
  --text: #000000;
  --border: #e0e0e0;
  --card-bg: #f9f9f9;
  --hover-bg: #f0f0f0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1e1e1e;
    --text: #e0e0e0;
    --border: #444;
    --card-bg: #2a2a2a;
    --hover-bg: #333;
  }
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
    Ubuntu, Cantarell, sans-serif;
  background: var(--bg);
  color: var(--text);
  margin: 0;
  padding: 2rem;
  line-height: 1.6;
}
.header {
  max-width: 1200px;
  margin: 0 auto 3rem;
}
.header h1 {
  margin: 0 0 1rem 0;
  font-size: 2.5rem;
}
.header p {
  margin: 0.5rem 0;
  font-size: 1.1rem;
}
.header a {
  color: #0066cc;
}
@media (prefers-color-scheme: dark) {
  .header a { color: #66b3ff; }
}
.section {
  max-width: 1200px;
  margin: 0 auto 4rem;
}
.section h2 {
  font-size: 1.8rem;
  margin: 2rem 0 1.5rem 0;
  border-bottom: 2px solid var(--border);
  padding-bottom: 0.5rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.5rem;
}
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
@media (prefers-color-scheme: dark) {
  .card:hover {
    box-shadow: 0 4px 12px rgba(255,255,255,0.1);
  }
}
.card-image {
  width: 100%;
  height: auto;
  display: block;
}
.card-meta {
  padding: 1rem;
}
.card-filename {
  font-family: 'Monaco', 'Courier New', monospace;
  font-size: 0.85rem;
  color: #666;
  margin: 0 0 0.5rem 0;
  word-break: break-all;
}
@media (prefers-color-scheme: dark) {
  .card-filename { color: #aaa; }
}
.card-caption {
  font-size: 0.95rem;
  margin: 0;
  line-height: 1.5;
}
.lightbox {
  display: none;
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.9);
  z-index: 1000;
  justify-content: center;
  align-items: center;
  overflow: auto;
  padding: 2rem;
}
.lightbox.active {
  display: flex;
}
.lightbox-content {
  position: relative;
  max-width: 90vw;
  max-height: 90vh;
}
.lightbox-image {
  max-width: 100%;
  max-height: 100%;
  display: block;
}
.lightbox-close {
  position: absolute;
  top: -40px;
  right: 0;
  background: none;
  border: none;
  color: #fff;
  font-size: 2rem;
  cursor: pointer;
  padding: 0 1rem;
  line-height: 1;
}
.lightbox-close:hover {
  color: #ccc;
}
""")

    html_parts.append("</style>")
    html_parts.append("</head>")
    html_parts.append("<body>")

    # Header
    html_parts.append('<div class="header">')
    html_parts.append("<h1>dynachaos — reproduction gallery</h1>")
    html_parts.append(
        "<p>Numerical reproductions of dynamical systems and chaos phenomena. "
        '<a href="https://github.com/openfluids/dynachaos">View on GitHub</a></p>'
    )
    html_parts.append("</div>")

    # Sections
    for section_id in SECTION_ORDER:
        spec = SECTION_SPECS[section_id]
        title = SECTION_TITLES[section_id]
        pngs = [f for f in spec.output_files if f.endswith(".png")]

        if not pngs:
            continue

        html_parts.append('<div class="section">')
        html_parts.append(f"<h2>{title}</h2>")
        html_parts.append('<div class="grid">')

        for png_name in pngs:
            caption_key = f"{section_id}/{png_name}"
            caption = CAPTIONS.get(caption_key, "")
            webp_name = png_name.replace(".png", ".webp")

            # Get thumbnail dimensions for layout shift prevention
            webp_path = site_thumbs_dir / section_id / webp_name
            thumb_width = None
            thumb_height = None
            if webp_path.exists():
                try:
                    thumb_img = Image.open(webp_path)
                    thumb_width, thumb_height = thumb_img.size
                except Exception:
                    pass

            data_full = f"full/{section_id}/{png_name}"
            html_parts.append(f'<div class="card" data-full="{data_full}">')
            if thumb_width and thumb_height:
                img_tag = (
                    f'<img class="card-image" '
                    f'src="thumbs/{section_id}/{webp_name}" '
                    f'loading="lazy" width="{thumb_width}" '
                    f'height="{thumb_height}" alt="{png_name}">'
                )
                html_parts.append(img_tag)
            else:
                img_tag = (
                    f'<img class="card-image" '
                    f'src="thumbs/{section_id}/{webp_name}" '
                    f'loading="lazy" alt="{png_name}">'
                )
                html_parts.append(img_tag)
            html_parts.append('<div class="card-meta">')
            html_parts.append(f'<div class="card-filename">{png_name}</div>')
            html_parts.append(f'<p class="card-caption">{caption}</p>')
            html_parts.append("</div>")
            html_parts.append("</div>")

        html_parts.append("</div>")
        html_parts.append("</div>")

    # Lightbox
    html_parts.append('<div id="lightbox" class="lightbox">')
    html_parts.append('<div class="lightbox-content">')
    html_parts.append('<button class="lightbox-close">×</button>')
    html_parts.append('<img id="lightbox-image" class="lightbox-image" src="" alt="">')
    html_parts.append("</div>")
    html_parts.append("</div>")

    # JavaScript
    html_parts.append("<script>")
    html_parts.append("""
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightbox-image');
const closeBtn = document.querySelector('.lightbox-close');

document.querySelectorAll('.card').forEach(card => {
  card.addEventListener('click', () => {
    const fullPath = card.getAttribute('data-full');
    lightboxImage.src = fullPath;
    lightbox.classList.add('active');
  });
});

closeBtn.addEventListener('click', () => {
  lightbox.classList.remove('active');
});

lightbox.addEventListener('click', (e) => {
  if (e.target === lightbox) {
    lightbox.classList.remove('active');
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    lightbox.classList.remove('active');
  }
});
""")
    html_parts.append("</script>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    html_content = "\n".join(html_parts)
    site_index = site_dir / "index.html"
    site_index.write_text(html_content)
    site_bytes = site_index.stat().st_size

    # Summary
    print(f"Thumbnails written: {thumb_count}")
    print(f"Thumbnail bytes: {thumb_bytes:,}")
    print(f"Site index bytes: {site_bytes:,}")
    print(f"Gallery built at: {site_dir / 'index.html'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
