"""Official-source, resumable download; no third-party copies."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener

BASE = "https://physionet.org/files/challenge-2019/1.0.0/"


class OfficialRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https" or urlparse(newurl).hostname != "physionet.org":
            raise ValueError("Refusing redirect outside official HTTPS PhysioNet")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_official(url, headers=None):
    if not url.startswith(BASE):
        raise ValueError("Only the versioned official PhysioNet source is allowed")
    return build_opener(OfficialRedirect()).open(Request(url, headers=headers or {}), timeout=60)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.files = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href", "")
        if tag == "a" and re.fullmatch(r"p[0-9]+\.psv", href):
            self.files.append(href)


def fetch_file(url, destination, checksum=None):
    destination = Path(destination)
    if destination.exists():
        if checksum and hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
            raise ValueError("Existing download checksum mismatch; refusing overwrite")
        from .data import read_psv
        read_psv(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".psv.part")
    offset = partial.stat().st_size if partial.exists() else 0
    with open_official(url, {"Range": f"bytes={offset}-"} if offset else {}) as response:
        resumed = offset and response.status == 206
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Invalid resume range")
        with partial.open("ab" if resumed else "wb") as handle:
            received = 0
            while block := response.read(1024 * 128):
                handle.write(block)
                received += len(block)
        expected = response.headers.get("Content-Length")
        if expected is not None and received != int(expected):
            raise ValueError("Incomplete HTTP response; partial download retained")
    if checksum and hashlib.sha256(partial.read_bytes()).hexdigest() != checksum:
        raise ValueError("Official checksum mismatch")
    from .data import read_psv
    read_psv(partial)
    partial.replace(destination)


def download(root, workers=12):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # The root checksum file is official; not all nested PSV files have published hashes.
    with open_official(BASE + "SHA256SUMS.txt") as response:
        checksums_text = response.read().decode()
    checksums = {}
    for line in checksums_text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            checksums[parts[1].lstrip("*")] = parts[0]
    (root / "official_SHA256SUMS.txt").write_text(checksums_text, encoding="utf-8")
    counts, tasks = {}, []
    for source in ("A", "B"):
        subdir = f"training/training_set{source}/"
        with open_official(BASE + subdir) as response:
            parser = Links()
            parser.feed(response.read().decode())
        names = sorted(set(parser.files))
        if not names:
            raise ValueError("No official PSV files discovered")
        counts[source] = len(names)
        for name in names:
            tasks.append((BASE + subdir + name, root / f"training_set{source}" / name, checksums.get(subdir + name)))

    def attempt(task):
        for retry in range(4):
            try:
                fetch_file(*task)
                return
            except (OSError, ValueError):
                if retry == 3:
                    raise
                time.sleep(2**retry)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(attempt, tasks), 1):
            if n % 1000 == 0:
                print(f"Validated {n}/{len(tasks)} files", flush=True)
    receipt = {"source_url": BASE, "download_date": datetime.now(timezone.utc).isoformat(),
               "listed_files": counts, "complete": True,
               "official_psv_checksums_available": sum(t[2] is not None for t in tasks),
               "validation": "official listing + HTTP length + full schema/parser; published checksums when available"}
    (root / "download_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
