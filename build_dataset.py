"""Build a real gopher vs not-gopher dataset from iNaturalist.

Positives: family Geomyidae (pocket gophers), taxon 44027.
Negatives: order Rodentia minus Geomyidae — squirrels, chipmunks, mice, rats,
voles, etc. Visually similar small mammals, so the task actually requires
learning gopher-specific features.

Output: data/gopher_dataset.npz with X_train/y_train/X_test/y_test, same layout
as the original file (N, 32, 32, 3) uint8, labels int32 in {0, 1}.
"""

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

API = "https://api.inaturalist.org/v1/observations"
IMG_SIZE = 32
PER_CLASS = 1000
OVERFETCH = 1.3  # pull extra URLs to survive download failures

GOPHER_TAXON = 44027
RODENTIA_TAXON = 43698

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "gopher-learning-project/0.1"})


def fetch_photo_urls(taxon_id, n_wanted, without_taxon_id=None):
    urls, seen = [], set()
    page = 1
    while len(urls) < n_wanted and page <= 50:
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",
            "per_page": 200,
            "page": page,
        }
        if without_taxon_id:
            params["without_taxon_id"] = without_taxon_id
        r = SESSION.get(API, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            break
        for obs in results:
            for photo in obs.get("photos", []):
                pid = photo["id"]
                if pid in seen:
                    continue
                seen.add(pid)
                urls.append(photo["url"].replace("/square.", "/small."))
                break  # one photo per observation
        page += 1
    return urls[:n_wanted]


def download_and_resize(url):
    try:
        r = SESSION.get(url, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)
        if arr.shape == (IMG_SIZE, IMG_SIZE, 3):
            return arr
    except Exception:
        pass
    return None


def gather(urls, label, desc):
    print(f"[{desc}] downloading {len(urls)} images...")
    arrs = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, arr in enumerate(ex.map(download_and_resize, urls)):
            if arr is not None:
                arrs.append(arr)
            if (i + 1) % 100 == 0:
                print(f"  [{desc}] {i + 1}/{len(urls)}  kept={len(arrs)}")
    X = np.stack(arrs)
    y = np.full(len(arrs), label, dtype=np.int32)
    print(f"[{desc}] final: {X.shape}")
    return X, y


def main():
    rng = np.random.default_rng(0)
    target = int(PER_CLASS * OVERFETCH)

    print("fetching URLs...")
    pos_urls = fetch_photo_urls(GOPHER_TAXON, n_wanted=target)
    neg_urls = fetch_photo_urls(
        RODENTIA_TAXON, n_wanted=target, without_taxon_id=GOPHER_TAXON
    )
    print(f"  got {len(pos_urls)} gopher urls, {len(neg_urls)} non-gopher urls")

    Xp, yp = gather(pos_urls, 1, "gopher")
    Xn, yn = gather(neg_urls, 0, "non-gopher")

    n = min(len(Xp), len(Xn), PER_CLASS)
    Xp, yp = Xp[:n], yp[:n]
    Xn, yn = Xn[:n], yn[:n]
    print(f"balanced to {n} per class")

    X = np.concatenate([Xp, Xn])
    y = np.concatenate([yp, yn])

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    out = "data/gopher_dataset.npz"
    np.savez(
        out, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    print(f"wrote {out}")
    print(f"  train: X={X_train.shape} labels={np.bincount(y_train).tolist()}")
    print(f"  test : X={X_test.shape} labels={np.bincount(y_test).tolist()}")


if __name__ == "__main__":
    main()
