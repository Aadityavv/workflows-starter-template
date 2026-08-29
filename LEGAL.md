# Legal and compliance

This is built into the product from day one, not bolted on later.

## What this tool operates on

**Your own files only.** `djmix` reads audio files you point it at. It has no
catalog search, no download feature, and no streaming integration. There is no
code path that acquires music.

This is a deliberate architectural choice, not just a legal one. DJ and remix
apps have repeatedly had streaming-platform API access revoked — Spotify has cut
off DJ apps before — so building the core product on catalog access would mean
building on a foundation that can be withdrawn without notice.

## If a streaming integration is added later

Any future integration must operate **only on the user's own saved library**,
through **their own authenticated session**, and must never become a general
catalog search or download feature. That constraint should be enforced in the
integration's design, not left to policy.

## Exported mixes

Exports are **for your personal use by default**. A merged mix containing music
you did not create is a derivative work.

Publishing, distributing, streaming, or monetising such a mix requires real
licensing — sync and mechanical licences from the relevant rights holders.
Nothing in this tool grants those rights, and no feature here should ever
silently enable public distribution. If a sharing or publishing feature is
built, it needs an explicit terms-of-service boundary and a clear, unmissable
warning to the user before the first share — not a checkbox buried in settings.

## Your data

- Analysis results are cached locally, keyed by file content hash. Nothing is
  uploaded.
- Audio is never sent to any LLM provider. Only *measured numbers* — BPM, key,
  timestamps, mood scores — are included in a planning prompt, and only when you
  explicitly choose `--planner llm`. The default planner makes no network calls
  at all.
- `djmix purge` deletes your analysis cache and rendered mixes. That is the
  delete-my-data path, and it exists now rather than being deferred.

## Test fixtures

Every audio file under `tests/fixtures/audio/` is synthesised from scratch by
`tests/fixtures/generate.py` — additive synthesis and seeded noise, no samples,
no recordings, no third-party material. Regenerate them with `make fixtures`.
