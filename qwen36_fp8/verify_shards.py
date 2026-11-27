import hashlib, json, os, sys, urllib.request
from concurrent.futures import ProcessPoolExecutor as E


def h(args):
    d, n = args
    x = hashlib.sha256()
    with open(os.path.join(d, n), "rb") as f:
        while b := f.read(1 << 24):
            x.update(b)
    return n, x.hexdigest()


def main():
    rev, d, out_json = sys.argv[1:4]
    ref_json = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("REF_SHA256", "")
    if ref_json and os.path.isfile(ref_json):
        exp = json.load(open(ref_json))
    else:
        m = json.load(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"https://huggingface.co/api/models/Logics-MLLM/Logics-SWE-Qwen3.6-27B/revision/{rev}?blobs=true",
                    headers={"User-Agent": "hf-check"},
                ),
                timeout=40,
            )
        )
        exp = {s["rfilename"]: s["lfs"]["sha256"] for s in m["siblings"] if s.get("lfs") and s["rfilename"].endswith(".safetensors")}
    with E(6) as e:
        r = dict(e.map(h, [(d, n) for n in exp]))
    bad = [n for n in exp if r[n] != exp[n]]
    print("checked", len(exp), "bad", bad)
    json.dump(r, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
