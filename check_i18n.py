"""Validate that every t("...") key used in src/ exists in all i18n JSON files."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
I18N = ROOT / "src" / "filefinder" / "assets" / "i18n"

# Literal keys: t("key") / t('key')
literal_re = re.compile(r"""\bt\(\s*["']([a-zA-Z0-9_.]+)["']""")
# f-string template keys: t(f"prefix{i + 1}_title") -> prefix1..5 etc.
fstring_re = re.compile(r"""\bt\(\s*f["']([a-zA-Z0-9_.]*?)\{[^}]*\}([a-zA-Z0-9_]*)["']""")

keys: set[str] = set()
for path in SRC.rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    keys.update(literal_re.findall(text))
    for prefix, suffix in fstring_re.findall(text):
        # expand the known slide range 1..5
        for i in range(1, 6):
            keys.add(f"{prefix}{i}{suffix}")

# Keys built indirectly via variables (state_key in settings_view)
keys.update(["settings.model_available", "settings.model_missing"])

langs = {}
for lang in ("zh", "uz", "en"):
    data = json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8"))
    langs[lang] = set(data)

# identical key sets?
sets = list(langs.values())
if sets[0] != sets[1] or sets[1] != sets[2]:
    for lang in ("zh", "uz", "en"):
        others = set().union(*(v for k, v in langs.items() if k != lang))
        missing = others - langs[lang]
        extra = langs[lang] - others
        if missing:
            print(f"{lang}: missing keys present elsewhere: {sorted(missing)}")
        if extra:
            print(f"{lang}: extra keys: {sorted(extra)}")
    sys.exit(1)

ok = True
for lang, available in langs.items():
    missing = sorted(k for k in keys if k not in available)
    if missing:
        ok = False
        print(f"{lang}: MISSING {len(missing)} used keys: {missing}")

unused = sorted(set().union(*langs.values()) - keys)
print(f"used keys: {len(keys)}; json keys per lang: {len(sets[0])}; unused json keys: {unused}")
if ok:
    print("I18N_OK")
else:
    sys.exit(1)
