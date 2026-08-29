# Architecture

## The rule everything else serves

> Every audio measurement — BPM, key, structure, energetic sections, timestamps —
> is computed from the waveform by DSP. An LLM never supplies, guesses, or looks
> up a measurement. It orders tracks, picks transition styles, and interprets
> natural language, strictly on top of measured values handed to it.

Prompt wording cannot enforce this. Four code-level gates do, in
`planning/validation.py`, cheapest first:

| Stage | Gate | Catches |
|---|---|---|
| 0 | Forbidden-field scan | any key matching `bpm`/`key`/`tempo`/`energy`/… at any depth |
| 1 | Strict Pydantic schema | wrong types, `"183.2"` as a string, `NaN`, extra keys |
| 2 | Graph integrity | broken chains, duplicates, out-of-range times, unheard tracks |
| 3 | Numeric provenance | timestamps and fade lengths that trace to no measurement |

The strongest gate is the least clever one: **`Transition` has no `bpm` and no
`key` field, and `extra="forbid"`.** Those categories of hallucination are
structurally impossible to express, not merely detected. Stage 3 defends the one
remaining numeric surface — timestamps.

`render()` accepts only a `ValidatedMixPlan`, and `validate_plan()` is the sole
place that type is constructed. There is no path from model output to audio that
skips validation.

### How strong Stage 3 actually is

Worth stating numerically rather than hand-waving. A timestamp is traceable if
it lands on the measured beat grid (±35 ms) *and* within 30 s of a structural
landmark. On the fixture set that accepts **~13%** of uniformly random
timestamps in normal mode and **~3%** in strict mode (downbeats only), which is
what `test_provenance_coverage_is_sparse` measures and pins.

An earlier draft also allowed free offsets from anchors — any multiple of 0.5 s
up to 10 s. That accepted 18–43% of random timestamps, which made the stage
close to decorative. Requiring the *conjunction* of grid-lock and anchor
proximity is what gives it teeth.

Two further mitigations matter more than the raw number: the LLM path defaults
to **strict** mode, and the prompt hands the model an explicit **menu** of legal
times computed from the grid, so a compliant model produces exact matches rather
than near-misses. The validator does not trust that menu — it re-derives
provenance from the analysis independently, so a bug in the menu builder cannot
launder an illegal number through validation.

## Module boundaries

```
audio/      measurement      real DSP, no LLM, no planning
planning/   ordering         rule planner, LLM planner, validation
render/     execution        deterministic, zero AI
llm/        providers        one interface: call_llm(prompt) -> str
```

`audio/`, `planning/`, and `render/` do not import each other; they communicate
only through the Pydantic models in `models.py`. That is what makes analysis and
rendering unit-testable with no LLM in the loop, and what lets providers be
swapped without touching audio code.

## Analysis

| Quantity | Method |
|---|---|
| Beat grid | HPSS → onset strength → `librosa.beat.beat_track` |
| BPM | robust least-squares fit over the beat times |
| Downbeats | 4/4 assumed; phase chosen by onset strength + chroma novelty, each z-scored and weighted by its own discriminability |
| Key | CENS chroma of the harmonic component → Krumhansl-Schmuckler correlation over 24 rotations |
| Structure | beat-synchronous recurrence matrix → Laplacian spectral clustering → label smoothing and merging |
| Energy | RMS curve, smoothed, peak window snapped to the grid |
| Mood | measured-feature heuristic (tempo, brightness, percussiveness, mode, dynamics) |

Two decisions are load-bearing:

**BPM comes from a regression over the beat times, not from librosa's returned
tempo and not from the median inter-beat interval.** At hop 512 / 22050 Hz a
frame is 23.2 ms, so a 120 BPM beat period is 21.5 frames and a median of
differences quantises to 21 or 22 — a 2.3% error. Regressing over the whole
sequence averages that out (0.00% error on the fixtures). It also guarantees
`bpm` and `beat_times` describe the *same* grid, which the validator depends on:
its bar-multiple check and its beat-snap check would otherwise contradict.

**There is no code path that returns 0.0 BPM.** Audio with no percussive
transients yields `ok=False` and `bpm=None`, and the analyzer refuses the track.
Refusing is correct: a track with no trackable beat cannot be beat-matched, and
inventing a tempo would be precisely the unmeasured number this design exists to
prevent.

One HPSS decomposition is shared across all four analyzers. Computing it per
module made a 90 s track take ~24 s instead of ~7 s.

## Rendering: tempo groups

The ordered plan is partitioned into runs whose tempos are within ±6% of a
common median. Inside a group every track is stretched by a single constant,
pitch-preserving factor, so the beat grids line up *exactly* and crossfades are
genuinely beat-locked. Between groups, a longer non-beatmatched blend is used
rather than mangling audio to force a match.

The alternative — ramping tempo through the fade — sounds smoother in principle,
but Rubber Band's offline API takes a constant factor, so a ramp means chunked
processing with seam discontinuities, and it destroys the reproducibility the
determinism test depends on.

## Honest limitations

- **Phrase alignment.** Beat grids align; musical 8- and 16-bar phrases may not.
  That needs better downbeat detection than the 4/4 phase heuristic.
- **Structure segmentation is the weakest analyzer**, and the synthetic fixtures
  flatter it: they repeat sections with identical harmony, which real
  arrangements do not. Expect worse chorus estimates on real, through-composed,
  or ambient material. The renderer never depends on the chorus estimate being
  right — it only shifts where a fade lands.
- **Mood tagging is a measured heuristic, not CLAP.** It obeys the no-LLM rule
  but is far cruder than the zero-shot audio-text model intended for Phase 2.
  The interface is the one CLAP will implement.
- **Tempo octave ambiguity is real and not a bug.** 87 and 174 BPM describe the
  same drum'n'bass groove. Both are recorded; the planner treats them as
  compatible rather than silently "correcting" one.
- **Determinism is same-machine.** Byte-identical output across machines is not
  achievable with Rubber Band, BLAS, and numba in the loop; the test asserts
  near-identity (< 1e-6) within a process.
- **Fixtures are synthesised.** They have exact ground truth and real percussive
  transients, but they are not a substitute for validating tolerances against
  real recordings. Drop real files into `tests/fixtures/audio/` and re-run.
