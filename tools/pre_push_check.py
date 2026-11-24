"""Pre-push check. Read-only: it reports problems and never changes any file.

Run from the project root, after `git init` and creating .gitignore:
    python tools/pre_push_check.py

It looks at the files Git would add (respecting .gitignore). If Git is not
available it falls back to walking the folder, skipping the usual excluded folders.
"""
import os
import subprocess
import sys

ROOT = os.getcwd()
WARN_MB, BLOCK_MB = 50, 100
TEXT_EXT = {".py", ".md", ".html", ".csv", ".json", ".txt", ".ipynb", ".yml", ".yaml", ".toml", ".cfg"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
USER = os.environ.get("USERNAME") or os.environ.get("USER") or os.path.basename(os.path.expanduser("~"))
NEEDLES = [USER, "C:\\\\Users", "C:\\Users", "c:\\Users", "c:\\\\Users", "C:/Users", "c:/Users"]
SELF_FILES = {os.path.join("tools", "pre_push_check.py"), os.path.join("tools", "sanitize_notebooks.py")}
REQUIRED = [
    "README.md", "requirements.txt", ".gitignore",
    os.path.join("app", "app.py"), os.path.join("app", "templates", "index.html"),
    os.path.join("src", "data.py"),
    os.path.join("models", "final_efficientnetb0.keras"),
    os.path.join("models", "final_config.json"),
    os.path.join("models", "ood_stats.npz"),
    os.path.join("data", "processed", "split_v1.csv"),
]
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".ipynb_checkpoints", ".vscode", "backups"}
SKIP_PATHS = {os.path.join("data", "raw"), os.path.join("data", "ood_check")}

problems, warnings = [], []


def candidate_files():
    """Files that would be uploaded."""
    if os.path.isdir(os.path.join(ROOT, ".git")):
        try:
            out = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=ROOT, capture_output=True, text=True, check=True).stdout
            files = [l.strip().replace("/", os.sep) for l in out.splitlines() if l.strip()]
            return files, "git (respects .gitignore)"
        except Exception as e:
            warnings.append(f"git ls-files failed ({type(e).__name__}); falling back to folder walk")
    files = []
    for dp, dirs, fs in os.walk(ROOT):
        rel = os.path.relpath(dp, ROOT)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                   and os.path.normpath(os.path.join(rel, d)) not in SKIP_PATHS]
        for f in fs:
            files.append(os.path.normpath(os.path.join(rel, f)))
    return files, "folder walk (does NOT respect .gitignore; run git init first for an exact list)"


files, mode = candidate_files()
fileset = set(os.path.normpath(f) for f in files)
print(f"Checking {len(files)} files using: {mode}\n")

# 1) required files
for r in REQUIRED:
    if not os.path.exists(os.path.join(ROOT, r)):
        problems.append(f"missing required file: {r}")
    elif os.path.normpath(r) not in fileset:
        problems.append(f"required file exists but would NOT be uploaded (ignored?): {r}")
if not os.path.exists(os.path.join(ROOT, "LICENSE")):
    warnings.append("no LICENSE file yet (your decision)")

# 2) sizes
print("Largest files that would be uploaded:")
sizes = []
for f in files:
    p = os.path.join(ROOT, f)
    if os.path.isfile(p):
        sizes.append((os.path.getsize(p) / 1e6, f))
for mb, f in sorted(sizes, reverse=True)[:8]:
    print(f"  {mb:8.1f} MB  {f}")
for mb, f in sizes:
    if mb >= BLOCK_MB:
        problems.append(f"{f} is {mb:.0f} MB: GitHub blocks files over {BLOCK_MB} MB (use Git LFS or Releases)")
    elif mb >= WARN_MB:
        warnings.append(f"{f} is {mb:.0f} MB: GitHub warns above {WARN_MB} MB (consider Git LFS)")
print()

# 3) other models, dataset or personal images
for f in files:
    fn = f.replace("\\", "/")
    if fn.startswith("models/") and fn.endswith((".keras", ".h5")) and not fn.endswith("final_efficientnetb0.keras"):
        problems.append(f"extra model file would be uploaded: {f}")
    if fn.startswith("data/") and os.path.splitext(fn)[1].lower() in IMG_EXT:
        problems.append(f"image under data/ would be uploaded (dataset or personal image): {f}")

# 4) personal paths in text files
hits = []
for f in files:
    if os.path.normpath(f) in SELF_FILES:
        continue
    if os.path.splitext(f)[1].lower() in TEXT_EXT:
        p = os.path.join(ROOT, f)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            continue
        n = sum(text.count(nd) for nd in NEEDLES)
        if n:
            hits.append((f, n))
for f, n in hits:
    warnings.append(f"personal path or username found {n}x in {f}")

# 5) split file header
sp = os.path.join(ROOT, "data", "processed", "split_v1.csv")
if os.path.exists(sp):
    with open(sp, "r", encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().strip()
        first = fh.readline().strip()
    print("split_v1.csv header:", header)
    if "abs" in header.lower() or ":\\" in first or ":/" in first:
        problems.append("split_v1.csv contains an absolute path column or value. "
                        "Do not edit the original; publish a separately named copy without it, "
                        "and make sure src/data.py can load it.")

print()
if warnings:
    print("WARNINGS")
    for w in warnings:
        print("  -", w)
if problems:
    print("PROBLEMS")
    for p in problems:
        print("  -", p)
if not problems:
    print("\nNo blocking problems found. Review the warnings above before pushing.")
sys.exit(1 if problems else 0)