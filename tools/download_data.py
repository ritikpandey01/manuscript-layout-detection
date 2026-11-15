#!/usr/bin/env python3
"""
Download manuscript page images for dev / gold / synthetic-background use.

Sources:
  1. archive.org  (advancedsearch + metadata + BookReader page images)
  2. Wikimedia Commons (API, gives per-file license info)

Outputs (relative to --out):
  raw/archive_org/*.jpg, raw/commons/*.jpg
  sources.csv        (source, id, page, url, license, title, local_file)
  contact_sheet_XX.jpg  (thumbnail grids, to pick header/footer/margin-note pages)

Usage:
  python download_data.py --out ./data --test            # tiny run: ~5 images
  python download_data.py --out ./data --target 100      # full run
Resumable: already-downloaded files are skipped.
Always check the license column in sources.csv before using an image.
"""
import argparse, csv, json, os, random, re, socket, sys, time
from pathlib import Path

import requests
import urllib3.util.connection as urllib_connection

# Force IPv4 to prevent IPv6 connection resets on Wikimedia / CDN endpoints
urllib_connection.allowed_gai_family = lambda: socket.AF_INET

# Comply with Wikimedia & archive.org robot policies (avoid generic example.com)
UA = {"User-Agent": "ManuscriptLayoutBot/1.0 (https://github.com/immverse/manuscript; contact: research@immverse.ai)"}
S = requests.Session()
S.headers.update(UA)

# Rich queries targeting diverse manuscript formats: palm leaf, paper, commentaries, margins, illuminations
ARCHIVE_QUERIES = [
    'subject:(palm leaf manuscript) AND mediatype:texts',
    'creator:(eGangotri) AND mediatype:texts',
    'title:(palm leaf) AND mediatype:texts',
    'subject:(sanskrit manuscript) AND mediatype:texts',
    'collection:(royalasiaticsociety) AND mediatype:texts',
    'subject:(devanagari manuscript) AND mediatype:texts',
    'subject:(grantha manuscript) AND mediatype:texts',
    'subject:(jain manuscript) AND mediatype:texts',
]
COMMONS_QUERIES = [
    "palm leaf manuscript",
    "Sanskrit manuscript page",
    "Devanagari manuscript",
    "Grantha manuscript",
    "Sharada manuscript",
    "Indian manuscript folio",
    "Sanskrit manuscript with commentary",
    "Jain manuscript",
    "Birch bark manuscript",
    "Newar manuscript",
]


def get_json(url, params=None, retries=3):
    for i in range(retries):
        try:
            r = S.get(url, params=params, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                print(f"  ! failed {url}: {e}")
                return None
            time.sleep(2 * (i + 1))


def download(url, dest: Path, retries=3):
    if dest.exists() and dest.stat().st_size > 5000:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") \
                    and len(r.content) > 5000:
                dest.write_bytes(r.content)
                return True
            if r.status_code in (403, 404):
                print(f"  ! HTTP {r.status_code} downloading {url}")
                return False
        except Exception as e:
            if i == retries - 1:
                print(f"  ! download error {url}: {e}")
        time.sleep(2 * (i + 1))
    return False


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


# ---------------- archive.org ----------------
def get_page_count(meta):
    m = meta.get("metadata", {})
    # 1. check imagecount in metadata
    if m.get("imagecount"):
        try:
            return int(m["imagecount"])
        except (ValueError, TypeError):
            pass
    # 2. check pages in metadata
    if m.get("pages"):
        try:
            return int(m["pages"])
        except (ValueError, TypeError):
            pass
    # 3. check Single Page Processed JP2 ZIP or _jp2.zip filecount
    for f in meta.get("files", []):
        if f.get("format") == "Single Page Processed JP2 ZIP" or "_jp2.zip" in f.get("name", ""):
            fc = f.get("filecount")
            if fc:
                try:
                    return int(fc)
                except (ValueError, TypeError):
                    pass
    # 4. check any file with filecount
    for f in meta.get("files", []):
        if "filecount" in f:
            try:
                cnt = int(f["filecount"])
                if cnt > 5:
                    return cnt
            except (ValueError, TypeError):
                pass
    return 0


def archive_items(query, rows=15):
    d = get_json("https://archive.org/advancedsearch.php", {
        "q": query, "fl[]": ["identifier", "title", "licenseurl"],
        "rows": rows, "output": "json", "sort[]": "downloads desc"})
    return (d or {}).get("response", {}).get("docs", [])


def archive_pages(item, n_pages, out_dir, rows, width=1400):
    ident = item["identifier"]
    meta = get_json(f"https://archive.org/metadata/{ident}")
    if not meta:
        return 0
    count = get_page_count(meta)
    if count < 1:
        # fallback: probe first pages
        count = 10
    lic = meta.get("metadata", {}).get("licenseurl") or item.get("licenseurl") or "unknown (check item page)"

    # sample evenly, skipping initial cover pages if possible
    if count >= 6:
        start_p = 2
        end_p = max(start_p + 1, count - 2)
        sample_pool = list(range(start_p, end_p))
    else:
        sample_pool = list(range(0, count))

    n_sample = min(n_pages, len(sample_pool))
    idx = sorted(set(random.sample(sample_pool, n_sample))) if n_sample > 0 else [0]
    got = 0
    for i in idx:
        url = f"https://archive.org/download/{ident}/page/n{i}_w{width}.jpg"
        dest = out_dir / f"{slug(ident)}_p{i}.jpg"
        if download(url, dest):
            rows.append(["archive.org", ident, i, url, lic, item.get("title", ""), str(dest)])
            got += 1
        time.sleep(0.5)
    return got


# ---------------- Wikimedia Commons ----------------
def commons_search(query, limit=40):
    d = get_json("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "generator": "search", "gsrsearch": query,
        "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata", "iiurlwidth": 1600, "format": "json"})
    pages = (d or {}).get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") not in ("image/jpeg", "image/png"):
            continue
        ext = ii.get("extmetadata", {})
        out.append({
            "title": p.get("title", ""),
            "url": ii.get("thumburl") or ii.get("url"),
            "license": ext.get("LicenseShortName", {}).get("value", "unknown"),
            "page": ii.get("descriptionurl", ""),
        })
    return out


# ---------------- contact sheet ----------------
def contact_sheets(files, out_dir, per_sheet=30, cols=6, thumb=300):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed; skipping contact sheets (pip install pillow)")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    created_sheets = []
    for s in range(0, len(files), per_sheet):
        chunk = files[s:s + per_sheet]
        rws = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rws * (thumb + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for k, f in enumerate(chunk):
            try:
                im = Image.open(f).convert("RGB")
                im.thumbnail((thumb - 4, thumb - 4))
                x, y = (k % cols) * thumb, (k // cols) * (thumb + 18)
                sheet.paste(im, (x + 2, y + 2))
                d.text((x + 3, y + thumb), Path(f).stem[-34:], fill="black")
            except Exception as e:
                print(f"  ! error processing {f} for contact sheet: {e}")
        sheet_path = out_dir / f"contact_sheet_{s // per_sheet + 1:02d}.jpg"
        sheet.save(sheet_path, quality=80)
        created_sheets.append(sheet_path)
    return created_sheets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data")
    ap.add_argument("--target", type=int, default=100, help="approx total images")
    ap.add_argument("--per-item", type=int, default=4, help="pages per archive.org item")
    ap.add_argument("--test", action="store_true", help="tiny run to check connectivity")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    if a.test:
        a.target, a.per_item = 5, 2

    out = Path(a.out)
    rows = []
    half = a.target // 2 or 1

    print("== archive.org ==")
    seen, total = set(), 0
    for q in ARCHIVE_QUERIES:
        if total >= half:
            break
        items = archive_items(q)
        for item in items:
            if total >= half:
                break
            if item["identifier"] in seen:
                continue
            seen.add(item["identifier"])
            n = archive_pages(item, a.per_item, out / "raw" / "archive_org", rows)
            total += n
            print(f"  {item['identifier']}: {n} pages (archive total: {total})")

    print("\n== Wikimedia Commons ==")
    ctotal = 0
    commons_seen = set()
    for q in COMMONS_QUERIES:
        if ctotal >= a.target - total:
            break
        results = commons_search(q)
        for r in results:
            if ctotal >= a.target - total:
                break
            if r["title"] in commons_seen:
                continue
            commons_seen.add(r["title"])
            dest = out / "raw" / "commons" / (slug(r["title"].replace("File:", "")) + ".jpg")
            if r["url"] and download(r["url"], dest):
                rows.append(["commons", r["title"], "", r["url"], r["license"], r["page"], str(dest)])
                ctotal += 1
                print(f"  {r['title'][:60]} [{r['license']}] (commons total: {ctotal})")
            time.sleep(0.5)

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sources.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["source", "id", "page", "url", "license", "title_or_page", "local_file"])
        w.writerows(rows)

    all_files = sorted(str(p) for p in (out / "raw").rglob("*.jpg"))
    sheets = contact_sheets(all_files, out)

    # Compute source counts
    archive_count = len(list((out / "raw" / "archive_org").glob("*.jpg"))) if (out / "raw" / "archive_org").exists() else 0
    commons_count = len(list((out / "raw" / "commons").glob("*.jpg"))) if (out / "raw" / "commons").exists() else 0

    print(f"\n================ SUMMARY ================")
    print(f"New images in this run: {len(rows)}")
    print(f"Total images on disk:   {len(all_files)}")
    print(f"  - archive.org:        {archive_count}")
    print(f"  - Wikimedia Commons:  {commons_count}")
    print(f"Sources metadata:       {csv_path}")
    print(f"Contact sheets created ({len(sheets)}):")
    for s in sheets:
        print(f"  - {s}")
    print(f"=========================================\n")

    if not all_files:
        print("No images downloaded. Check internet access / try --test and read errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
