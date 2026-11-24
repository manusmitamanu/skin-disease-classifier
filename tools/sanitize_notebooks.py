"""Replace your local project path and Windows username inside notebook files.

Run from the project root:
    python tools/sanitize_notebooks.py

What it does:
  1. Copies every notebook in notebooks/ to backups/notebooks_before_sanitize/ (git-ignored).
  2. In notebooks/*.ipynb, replaces the full project path with <project> and any
     remaining C:\\Users\\<your username> with <user-home>.
  3. Re-opens each changed notebook as JSON to make sure it is still valid.
It never touches code logic beyond those text replacements, and it stops
without changing a notebook if the result is not valid JSON.
"""
import glob
import json
import os
import re
import shutil

ROOT = os.getcwd()
NB_DIR = os.path.join(ROOT, "notebooks")
BACKUP = os.path.join(ROOT, "backups", "notebooks_before_sanitize")
USER = os.environ.get("USERNAME") or os.environ.get("USER") or os.path.basename(os.path.expanduser("~"))
PROJECT = os.path.basename(ROOT)

SEP = r"(?:\\\\|\\|/)+"   # separators as they can appear inside JSON text
project_re = re.compile(
    rf"[A-Za-z]:{SEP}Users{SEP}{re.escape(USER)}{SEP}(?:[^\\/\"]+{SEP})*?{re.escape(PROJECT)}",
    re.IGNORECASE)
home_re = re.compile(rf"[A-Za-z]:{SEP}Users{SEP}{re.escape(USER)}", re.IGNORECASE)

os.makedirs(BACKUP, exist_ok=True)
paths = sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb")))
print(f"{len(paths)} notebooks found; user='{USER}', project folder='{PROJECT}'")

for p in paths:
    name = os.path.basename(p)
    shutil.copy2(p, os.path.join(BACKUP, name))
    with open(p, "r", encoding="utf-8") as fh:
        text = fh.read()
    new, n1 = project_re.subn("<project>", text)
    new, n2 = home_re.subn("<user-home>", new)
    try:
        json.loads(new)
    except Exception as e:
        print(f"SKIPPED {name}: result is not valid JSON ({type(e).__name__}); file left unchanged")
        continue
    if new != text:
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(new)
    print(f"{name}: {n1} project paths, {n2} other home paths replaced")

print("\nBackups are in:", BACKUP)
print("Now re-run: python tools/pre_push_check.py")