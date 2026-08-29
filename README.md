# djmix

Turn your own music files into seamless, DJ-style merged mixes — grouped by
occasion, or described in plain language.

```bash
djmix analyze ~/Music/library
djmix mix ~/Music/library --occasion workout --out workout.wav
djmix mix ~/Music/library --prompt "a 20-minute high-energy pre-workout mix" \
          --planner llm --duration 20m --out pump.wav
```

## The rule this project is built around

**Every audio measurement is computed from the waveform. An LLM never supplies
one.**

BPM, musical key, song structure, and energy curves come from real DSP
(librosa). The LLM's job is the creative layer only: reading your request,
choosing a running order, picking transition styles. It operates strictly on top
of numbers that were measured for it.

This is enforced in code, not in the prompt. The mix-plan schema has **no field**
for a BPM or a key, so those hallucinations cannot be expressed. Every timestamp
a planner does emit must trace back to a measured beat, downbeat, or structural
landmark, or the plan is rejected and retried, and then the deterministic
rule-based planner takes over. See [ARCHITECTURE.md](ARCHITECTURE.md) for how
strong that check actually is — measured, not asserted.

## Install

```bash
pip install -e ".[dev,llm]"
```

Python 3.11+. No system packages required for wav/flac/mp3/ogg — `soundfile`
bundles libsndfile. `ffmpeg` is needed only for m4a/aac, and is picked up from
your PATH or from the `imageio-ffmpeg` wheel automatically.

## How it works

1. **Analyze** — each file is measured once and cached by content hash: beat
   grid, BPM, key (Camelot too), structural segments, chorus estimate,
   peak-energy window, and a mood distribution. About 7 s per track; re-runs are
   instant.
2. **Plan** — tracks are ordered along an energy arc chosen for the occasion,
   respecting tempo continuity and harmonic compatibility. Each track
   contributes its strongest measured section rather than its full runtime, so a
   duration target is actually reachable.
3. **Validate** — the plan passes four gates before anything is rendered.
4. **Render** — deterministic and AI-free: pitch-preserving time-stretch into
   tempo groups, beat-aligned equal-power crossfades, and a −14 LUFS mastering
   pass with true-peak limiting.

## Commands

| Command | What it does |
|---|---|
| `djmix analyze PATH` | Measure a file or a whole directory |
| `djmix inspect TRACK` | Every measured value for one track |
| `djmix mix LIBRARY` | Plan and render (`--occasion`, `--prompt`, `--duration`, `--planner`) |
| `djmix validate PLAN` | Check a plan and print where every number came from |
| `djmix purge` | Delete your cache and rendered mixes |
| `djmix tiers` | Show configured entitlement limits |

Occasions ship in `config/occasions.yaml` — road trip, workout, party, romantic,
spiritual, focus, dinner. Adding one is a config edit, not a code change.

`djmix validate` is the interesting one:

```
VALID
 field                        value    rule                  anchor            delta
 plan[0].transition.out_at   51.734   SNAP_DOWNBEAT+ANCHOR   downbeat_grid    -0.000
 plan[0].transition.len_sec   5.517   BAR_LENGTH             2.0bar@87.00bpm  -0.000
```

## LLM providers

Pluggable behind a single `call_llm(prompt) -> str` interface: **Groq**
(default), **Anthropic**, and a deterministic **mock**.

```bash
export DJMIX_LLM_PROVIDER=groq GROQ_API_KEY=...
djmix mix ~/Music --prompt "something calm for a long drive" --planner llm
```

`--planner rule` is the default and makes **no network calls at all**. The LLM
path always degrades to it rather than failing, and says so when it does. The
whole test suite runs offline against the mock, with the network blocked.

Only measured numbers are ever sent to a provider. Your audio never is.

## Testing

```bash
make test        # full suite, offline, no API key needed
make fixtures    # regenerate the ground-truth test audio
make demo        # analyze -> mix -> validate end to end
```

Test audio is synthesised with exactly known tempo, key, and section layout —
a real kit (pitch-swept kick with a click transient, noise snare, hats) over a
harmonic bed. **Not** sine tones: a steady tone has no recurring attack
transients, gives the onset detector a flat envelope, and produces a false
`0.0 BPM` reading. A test asserts the fixtures have real onsets *before* any
tempo test runs, so a fixture regression can't be misread as an analyzer bug.

## Status and limitations

Phases 0 and 1 of a larger product: measurement, planning, validation, and
rendering, as a CLI. No web UI, job queue, or accounts yet — but the module
boundaries are drawn so those attach without touching audio code.

Read the **Honest limitations** section of [ARCHITECTURE.md](ARCHITECTURE.md)
before trusting output on real music. In short: transitions are beat-locked but
not always phrase-perfect; structure segmentation is the weakest analyzer and
the synthetic fixtures flatter it; mood tagging is a measured heuristic standing
in for CLAP.

## Legal

Your own files only. Exports are for personal use. See [LEGAL.md](LEGAL.md) —
it is short, and it is worth reading before you share anything.
