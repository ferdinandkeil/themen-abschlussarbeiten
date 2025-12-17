#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import html
import shutil
import pathlib
import subprocess
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.parse


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
ASSETS = ROOT / "assets"
BUILD = ROOT / "build"

TOPIC_LABEL = os.getenv("TOPIC_LABEL", "topic")  # label that marks topic issues
SITE_TITLE = os.getenv("SITE_TITLE", "Abschlussarbeitsthemen")
BASE_PATH = os.getenv("BASE_PATH", "/")  # for GH Pages usually "/"
LANG = os.getenv("LANGUAGE", "de")

GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")  # owner/repo
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # provided by actions


@dataclass
class Topic:
    number: int
    title: str
    slug: str
    url: str  # github html url
    degree: str = ""
    status: str = ""
    tags: List[str] = field(default_factory=list)
    start: str = ""
    prereq: str = ""
    body_markdown: str = ""
    closed: bool = False


def _die(msg: str) -> None:
    raise SystemExit(msg)


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s or "thema"


def run_pandoc(input_md: pathlib.Path, output_html: pathlib.Path, template_html: pathlib.Path, title: str) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "pandoc",
        str(input_md),
        "-f", "markdown+yaml_metadata_block",
        "-t", "html5",
        "--template", str(template_html),
        "-V", f"lang={LANG}",
        "-V", f"title={title}",
        "-o", str(output_html),
    ])


def http_get_json(url: str, token: Optional[str]) -> Tuple[Any, Dict[str, str]]:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "topics-site-builder")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
        headers = {k.lower(): v for k, v in resp.headers.items()}
    return json.loads(raw), headers


def parse_link_header(link_header: str) -> Dict[str, str]:
    # Example:
    # <https://api.github.com/...&page=2>; rel="next", <...&page=4>; rel="last"
    out: Dict[str, str] = {}
    for part in link_header.split(","):
        part = part.strip()
        m = re.match(r'<([^>]+)>\s*;\s*rel="([^"]+)"', part)
        if m:
            out[m.group(2)] = m.group(1)
    return out


def fetch_issues(repo: str, label: str, token: Optional[str]) -> List[dict]:
    issues: List[dict] = []
    url = f"https://api.github.com/repos/{repo}/issues?state=all&labels={urllib.parse.quote(label)}&per_page=100"
    while url:
        data, headers = http_get_json(url, token)
        if not isinstance(data, list):
            _die(f"GitHub API returned unexpected response for {url}")
        issues.extend(data)
        link = headers.get("link", "")
        links = parse_link_header(link) if link else {}
        url = links.get("next")
    return issues


def labels_to_fields(labels: List[dict]) -> Tuple[str, str, List[str]]:
    degree = ""
    status = ""
    tags: List[str] = []
    for lab in labels:
        name = (lab.get("name") or "").strip()
        low = name.lower()

        # degree labels like degree:BSc / degree:MSc
        if low.startswith("degree:"):
            degree = name.split(":", 1)[1].strip()

        # status labels like status:open/status:taken/status:draft
        if low.startswith("status:"):
            status = name.split(":", 1)[1].strip()

        # tags like tag:embedded
        if low.startswith("tag:"):
            t = name.split(":", 1)[1].strip()
            if t:
                tags.append(t)

    # normalization
    degree = degree.replace("bsc", "BSc").replace("msc", "MSc")
    status = status.lower()
    return degree, status, sorted(set(tags), key=str.lower)


def split_frontmatter(md: str) -> Tuple[Dict[str, Any], str]:
    """
    If body starts with YAML frontmatter:
    ---
    key: value
    ---
    returns (meta, rest)
    """
    md = md.lstrip("\ufeff")  # BOM safety
    s = md.lstrip()
    if not s.startswith("---"):
        return {}, md

    lines = s.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, md

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, md

    fm_text = "\n".join(lines[1:end_idx]).strip()
    rest = "\n".join(lines[end_idx + 1:]).lstrip()

    meta: Dict[str, Any] = {}
    # Try PyYAML if available, else fallback
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(fm_text) if fm_text else {}
        if isinstance(loaded, dict):
            meta = loaded
    except Exception:
        for ln in fm_text.splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")

    return meta, rest


def merge_meta(topic: Topic, meta: Dict[str, Any]) -> None:
    def get_str(key: str) -> str:
        v = meta.get(key)
        return str(v).strip() if v is not None else ""

    degree = get_str("degree")
    status = get_str("status")
    start = get_str("start")
    prereq = get_str("prereq")

    tags_val = meta.get("tags")
    tags: List[str] = []
    if isinstance(tags_val, list):
        tags = [str(x).strip() for x in tags_val if str(x).strip()]
    elif isinstance(tags_val, str) and tags_val.strip():
        tags = [t.strip() for t in tags_val.split(",") if t.strip()]

    if degree:
        topic.degree = degree
    if status:
        topic.status = status.lower()
    if start:
        topic.start = start
    if prereq:
        topic.prereq = prereq
    if tags:
        topic.tags = sorted(set(topic.tags + tags), key=str.lower)


def topic_is_open(topic: Topic) -> bool:
    # Closed issues are archive unless explicitly status:open
    if topic.closed and topic.status != "open":
        return False
    return topic.status == "open"


def ensure_templates() -> Tuple[pathlib.Path, pathlib.Path]:
    theme = TEMPLATES / "theme.html"
    index = TEMPLATES / "index.html"
    if not theme.exists():
        _die(f"Missing template: {theme}")
    if not index.exists():
        _die(f"Missing template: {index}")
    return theme, index


def copy_assets() -> None:
    if not ASSETS.exists():
        return
    dst = BUILD / "assets"
    dst.mkdir(parents=True, exist_ok=True)
    for p in ASSETS.rglob("*"):
        if p.is_file():
            rel = p.relative_to(ASSETS)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def write_topic_markdown(topic: Topic, out_md: pathlib.Path) -> None:
    """
    Create a markdown file that pandoc will render with template.
    Includes a small YAML metadata header for pandoc ($title$).
    """
    out_md.parent.mkdir(parents=True, exist_ok=True)

    tags_str = ", ".join([f'"{t.replace(chr(34), chr(39))}"' for t in topic.tags])
    tags_pretty = ", ".join(topic.tags)

    meta_lines = [
        "---",
        f'title: "{topic.title.replace('"', '\'')}"',
        f'number: {topic.number}',
        f'degree: "{topic.degree}"',
        f'status: "{topic.status}"',
        f'tags: [{tags_str}]',
        f'github: "{topic.url}"',
        "---",
        "",
    ]
    header = "\n".join(meta_lines)

    # Build a meta info box as HTML (safe-escaped)
    info_bits: List[str] = []
    if topic.degree:
        info_bits.append(f"<strong>Abschluss:</strong> {html.escape(topic.degree)}")
    if topic.status:
        info_bits.append(f"<strong>Status:</strong> {html.escape(topic.status)}")
    if topic.start:
        info_bits.append(f"<strong>Start:</strong> {html.escape(topic.start)}")
    if topic.prereq:
        info_bits.append(f"<strong>Voraussetzungen:</strong> {html.escape(topic.prereq)}")
    if tags_pretty:
        info_bits.append(f"<strong>Tags:</strong> {html.escape(tags_pretty)}")

    # GitHub link is markdown (Pandoc turns it into <a>)
    info_bits.append(f"[Auf GitHub ansehen]({topic.url})")

    # CSS-Klassen konsistent zum Index (<li class="status-open degree-bsc ...">)
    classes: List[str] = ["topic-meta-card"]

    status_cls = slugify((topic.status or "").strip().lower())
    if status_cls:
        classes.append(f"status-{status_cls}")

    degree_cls = slugify((topic.degree or "").strip().lower())
    if degree_cls:
        classes.append(f"degree-{degree_cls}")

    # Optional: Tag-Klassen auch auf der Detailseite (praktisch für spätere Filter/Badges)
    for t in topic.tags:
        tt = slugify((t or "").strip().lower())
        if tt:
            classes.append(f"tag-{tt}")

    info_block = (
        f'<div class="{" ".join(classes)}">\n'
        + "<br>\n".join(info_bits)
        + "\n</div>"
    )

    body = (topic.body_markdown or "").strip()
    full = f"{header}\n# {topic.title}\n\n{info_block}\n\n---\n\n{body}\n"
    out_md.write_text(full, encoding="utf-8")


def topic_list_item(topic: Topic) -> str:
    href = f"{BASE_PATH.rstrip('/')}/topics/{topic.number}-{topic.slug}.html"

    classes: List[str] = []

    # status-* (open/taken/draft/...)
    status = (topic.status or "").strip().lower()
    if status:
        classes.append(f"status-{slugify(status)}")

    # degree-* (bsc/msc/...)
    degree = (topic.degree or "").strip().lower()
    if degree:
        classes.append(f"degree-{slugify(degree)}")

    # Optional: tag-* Klassen (wenn du später per CSS/JS filtern willst)
    # Achtung: kann viele Klassen erzeugen; wenn du das nicht willst, einfach auskommentiert lassen.
    for t in topic.tags:
        tt = (t or "").strip().lower()
        if tt:
            classes.append(f"tag-{slugify(tt)}")

    class_attr = f' class="{" ".join(classes)}"' if classes else ""

    # Meta-Text (wie bisher), nur ohne Klammern geht es oft cleaner:
    bits = []
    if topic.degree:
        bits.append(topic.degree)
    if topic.status:
        bits.append("offen" if topic.status == "open" else ("vergeben" if topic.status == "taken" else topic.status))
    if topic.tags:
        bits.append(", ".join(topic.tags))
    meta = " · ".join(bits)
    meta_html = f' <small>{html.escape(meta)}</small>' if meta else ""

    return f'<li{class_attr}><a href="{href}">{html.escape(topic.title)}</a>{meta_html}</li>'


def build_index(topics: List[Topic], index_template: pathlib.Path) -> None:
    open_items = [topic_list_item(t) for t in topics if topic_is_open(t)]
    other_items = [topic_list_item(t) for t in topics if not topic_is_open(t)]

    def ul(items: List[str]) -> str:
        if not items:
            return "<p><em>Keine Einträge.</em></p>"
        return "<ul>\n" + "\n".join(items) + "\n</ul>\n"

    tpl = index_template.read_text(encoding="utf-8")
    out = (tpl
           .replace("$site_title$", html.escape(SITE_TITLE))
           .replace("$generated_at$", date.today().isoformat())
           .replace("$topics_open_html$", ul(open_items))
           .replace("$topics_other_html$", ul(other_items))
           )
    (BUILD / "index.html").write_text(out, encoding="utf-8")


def main() -> None:
    if not GITHUB_REPOSITORY:
        _die("GITHUB_REPOSITORY is not set (expected 'owner/repo').")
    if not GITHUB_TOKEN:
        _die("GITHUB_TOKEN is not set. In Actions, pass env: GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}")

    theme_tpl, index_tpl = ensure_templates()

    # fresh build dir
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True, exist_ok=True)
    copy_assets()

    raw = fetch_issues(GITHUB_REPOSITORY, TOPIC_LABEL, GITHUB_TOKEN)

    topics: List[Topic] = []
    for it in raw:
        # filter PRs (GitHub returns PRs in /issues endpoint)
        if "pull_request" in it:
            continue

        number = int(it.get("number"))
        title = (it.get("title") or "").strip()
        if not title:
            continue

        labels = it.get("labels") or []
        degree, status, tags = labels_to_fields(labels)

        body = it.get("body") or ""
        meta, body_rest = split_frontmatter(body)

        closed = bool(it.get("closed_at"))
        # default status if not present:
        # open issue -> open, closed issue -> taken
        default_status = "open" if not closed else "taken"

        t = Topic(
            number=number,
            title=title,
            slug=slugify(title),
            url=it.get("html_url") or "",
            degree=degree,
            status=status or default_status,
            tags=tags,
            body_markdown=body_rest,
            closed=closed,
        )

        if isinstance(meta, dict) and meta:
            merge_meta(t, meta)

        topics.append(t)

    # Sort: open first, then degree, then title
    def sort_key(t: Topic) -> Tuple[int, str, str]:
        open_rank = 0 if topic_is_open(t) else 1
        deg = (t.degree or "").lower()
        return (open_rank, deg, t.title.lower())

    topics.sort(key=sort_key)

    # Render each topic
    for t in topics:
        md_path = BUILD / "md" / f"{t.number}-{t.slug}.md"
        html_path = BUILD / "topics" / f"{t.number}-{t.slug}.html"
        write_topic_markdown(t, md_path)
        run_pandoc(md_path, html_path, theme_tpl, t.title)

    # Build index
    build_index(topics, index_tpl)

    print(f"Built {len(topics)} topics into {BUILD}")


if __name__ == "__main__":
    main()

