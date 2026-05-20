"""Download the Cityscapes packages required by the cutout pipeline.

Credentials are read from the ``CITYSCAPES_USERNAME`` and ``CITYSCAPES_PASSWORD``
environment variables; register at https://www.cityscapes-dataset.com/ first.
Packages are extracted into ``cfg.dataset_root`` by default, giving the
``leftImg8bit/{train,val,test}`` and ``gtFine/{train,val,test}`` layout that
``generate_dataset.py`` expects.

The login + ``packageID`` endpoints used here are the same unofficial routes
every Cityscapes downloader (HuggingFace, mmcv, etc.) relies on. They have
been stable for years but are not a documented API.
"""

__author__ = "RocketFlash: https://github.com/RocketFlash"

import argparse
import http.cookiejar
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

try:
    from .config import data_generation as cfg
except ImportError:
    from config import data_generation as cfg

BASE_URL = "https://www.cityscapes-dataset.com"
LOGIN_URL = f"{BASE_URL}/login/"
DOWNLOAD_URL = BASE_URL + "/file-handling/?packageID={package_id}"

# name -> (packageID, filename, top-level extracted dir)
PACKAGES = {
    "gtFine": (1, "gtFine_trainvaltest.zip", "gtFine"),
    "leftImg8bit": (3, "leftImg8bit_trainvaltest.zip", "leftImg8bit"),
}
DEFAULT_PACKAGES = ["leftImg8bit", "gtFine"]


def make_opener():
    """Return a urllib opener that persists cookies across requests."""
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def login(opener, username, password):
    """POST credentials to Cityscapes; raise RuntimeError on failure."""
    data = urllib.parse.urlencode(
        {"username": username, "password": password, "submit": "Login"}
    ).encode()
    response = opener.open(LOGIN_URL, data=data, timeout=30)
    body = response.read()
    # Bad credentials -> server re-renders the login form (HTTP 200, no redirect).
    if b'name="password"' in body:
        raise RuntimeError(
            "Cityscapes login failed: server returned the login form. "
            "Check CITYSCAPES_USERNAME / CITYSCAPES_PASSWORD."
        )


def download_package(opener, package_id, dest_path):
    """Stream a Cityscapes package to dest_path with a progress bar."""
    url = DOWNLOAD_URL.format(package_id=package_id)
    response = opener.open(url, timeout=120)
    # Authed requests for a real package return application/zip; an HTML body
    # here means the session is unauthenticated or not entitled to the package.
    ctype = response.headers.get("Content-Type", "")
    if "text/html" in ctype:
        raise RuntimeError(
            f"Expected zip for packageID={package_id}, got HTML. "
            "Login may have failed silently, or your account is not entitled "
            "to this package."
        )

    total = int(response.headers.get("Content-Length", "0")) or None
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    chunk = 1024 * 1024
    with (
        open(tmp_path, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as bar,
    ):
        while True:
            buf = response.read(chunk)
            if not buf:
                break
            f.write(buf)
            bar.update(len(buf))
    tmp_path.rename(dest_path)


def extract(zip_path, target_dir):
    """Extract zip into target_dir, skipping files that already exist."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for name in tqdm(members, desc=f"extract {zip_path.name}", unit="file"):
            out = target_dir / name
            if out.exists() and not name.endswith("/"):
                continue
            zf.extract(name, target_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Credentials must be supplied via the CITYSCAPES_USERNAME and "
            "CITYSCAPES_PASSWORD environment variables.\n"
            "Register first at https://www.cityscapes-dataset.com/."
        ),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=cfg.dataset_root,
        help="where to extract the packages (default: %(default)s)",
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        choices=sorted(PACKAGES),
        default=DEFAULT_PACKAGES,
        help="which packages to fetch (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="keep the downloaded .zip files after extraction",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a package if its top-level directory already has content",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    username = os.environ.get("CITYSCAPES_USERNAME")
    password = os.environ.get("CITYSCAPES_PASSWORD")
    if not username or not password:
        sys.exit(
            "Set CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD environment "
            "variables before running this script."
        )

    target_dir = args.target_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    opener = make_opener()
    login(opener, username, password)

    for name in args.packages:
        package_id, filename, top_dir_name = PACKAGES[name]
        top_dir = target_dir / top_dir_name
        if args.skip_existing and top_dir.exists() and any(top_dir.iterdir()):
            print(f"[skip] {name}: {top_dir} already populated")
            continue

        zip_path = target_dir / filename
        if zip_path.exists():
            print(f"[skip download] {filename} already present")
        else:
            print(f"[download] {filename}")
            download_package(opener, package_id, zip_path)

        print(f"[extract] {filename} -> {target_dir}")
        extract(zip_path, target_dir)

        if not args.keep_zips:
            zip_path.unlink()
            print(f"[cleanup] removed {filename}")

    print(f"done. Cityscapes data at: {target_dir}")


if __name__ == "__main__":
    main()
