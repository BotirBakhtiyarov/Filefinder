# Public / Private repo split

FileFinder ships in two repositories:

| Repo | Visibility | Contents |
|---|---|---|
| `Filefinder` | **Public** | Full free-tier app, license *verification* logic (without the secret), placeholder built-in key |
| `FileFinder-pro` (yaratiladi) | **Private** | Everything, including the private files below |

## Private files (never commit to the public repo)

- `src/filefinder/core/_license_secret.py` — HMAC secret for license keys (gitignored)
- `src/filefinder/core/builtin_key.py` — real built-in DeepSeek key blob (public copy gets an empty placeholder)
- `tools/generate_keys.py` — admin key generator
- `tools/embed_llm_key.py` — admin key embedding tool
- `keys*.csv` — issued key registry

## Publishing the public repo

```bash
python tools/publish_public.py --target ../FileFinder-public
cd ../FileFinder-public
git init && git remote add origin <public-repo-url>
git add -A && git commit -m "v3.0.0" && git push -u origin main
```

The script strips every private file and replaces the built-in key blob
with an empty placeholder. Public self-builds run as Free tier and do NOT
accept sold Pro keys (they fall back to the dev secret) — by design.

## Security notes (honest assessment)

1. **Key forging**: without `_license_secret.py` the HMAC checksum cannot be
   reproduced, so keys cannot be forged from the public source. The secret
   also ships (obfuscated) inside official builds — a determined attacker
   with the compiled app could extract it. This is inherent to offline
   licensing; the split raises the bar to "not worth the effort" for a
   ¥19.9/month app.
2. **Gate removal**: anyone can fork the public repo and delete the Pro
   checks. Unavoidable for open source. The practical protection is that
   the *built-in DeepSeek key* is only in official builds — forks get no
   free cloud LLM.
3. **Secret backup**: keep `_license_secret.py` backed up (password
   manager / private drive). Losing it invalidates all issued keys.
4. **Blacklist**: revoke abused keys by adding them to
   `src/filefinder/assets/blacklist.json` in the next release.
