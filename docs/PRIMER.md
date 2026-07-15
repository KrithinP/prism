# PRISM — The Primer

*Read this before `LITERATURE.md`. That document is a reference written for someone already
inside this subfield. This one builds the ideas from the ground up, in the order you actually
need them, with real numbers from our own setup.*

You know PyTorch, CNNs, and backprop. This primer does **not** re-teach those. What it teaches is
the vocabulary and the machinery of **object detection** and **sensor fusion** — a different world
with its own metrics, its own failure modes, and its own folklore. That's the gap, and it's a
completely normal gap. Nobody absorbs it by osmosis.

There is **one central idea** in this whole project. Everything below is scaffolding for it, and
I'll flag it when we get there (Part 3). If you only truly understand one thing, make it that.

---

## Part 0 — What we are actually building, in one paragraph

We have two cameras looking at the same scene: an ordinary **RGB** camera and a **thermal** (LWIR)
camera. Each produces an image. We want one neural network that looks at **both** images together
and draws boxes around every pedestrian. The interesting question — the entire research question —
is **where inside the network the two streams should meet, and what operation should merge them.**
Merge too early and you force one set of filters to serve two very different kinds of image. Merge
too late and the streams never help each other see. Merge in the middle — "middle fusion" — and
you supposedly get the best of both. Our job is to find out whether that's actually true, and
under what conditions it stops being true.

---

## Part 1 — What a detector is, and what it actually outputs

A **classifier** answers "what is in this image?" → one label.

A **detector** answers "what is in this image, *and where*?" → a **list** of predictions, each one:

```
(x, y, w, h,  class,  confidence)
 └── box ──┘  person   0.87
```

The output is a *variable-length list*. That's the whole difficulty. A classifier has a fixed
output shape; a detector might find zero pedestrians or thirty, and it doesn't know in advance.

**How YOLOX handles this.** It doesn't predict a list directly. It divides the image into a grid
and, at **every grid cell**, predicts "is there an object centred near me, and if so, what box?"
Most cells say "no" (low confidence). A few say "yes". You then throw away the low-confidence ones
and merge duplicates. That's it — that's the whole trick of one-stage detection.

Two words you'll see constantly:

- **Anchor-free** — at each cell, predict the box directly. YOLOX is anchor-free.
  (The older "anchor-based" way: pre-define a set of template box shapes at each cell and predict
  *offsets* from them. It works, but you have to hand-tune the templates to your data. Anchor-free
  drops that whole nuisance — which is a real gift for us, because fusion changes feature
  statistics and we'd otherwise have to re-tune anchors for every ablation.)
- **NMS (Non-Maximum Suppression)** — the duplicate-merging step. Ten neighbouring cells all fire
  on the same pedestrian; NMS keeps the most confident and deletes the rest.

**Confidence is the key to everything in Part 2.** Every prediction carries a score in [0, 1]. You
choose a **threshold**: keep predictions above it, discard the rest. Set the threshold low → you
catch nearly every pedestrian but also hallucinate a lot of false ones. Set it high → almost
everything you report is real, but you miss people. **You cannot escape this trade-off; you can
only choose where to sit on it.** A detector isn't one thing — it's a whole *curve* of possible
behaviours, one per threshold.

Hold that thought.

---

## Part 2 — How you grade a detector (and why KAIST doesn't use mAP)

### 2.1 Was this box right? — IoU

You predicted a box. There's a ground-truth box. Did you get it?

**IoU (Intersection over Union)** = (area of overlap) / (area of union). It's 1.0 for a perfect
match, 0.0 for no overlap. The convention: **IoU ≥ 0.5 counts as a hit.**

So for one image, at one confidence threshold, each prediction is either:
- **TP** (true positive) — matched a real pedestrian at IoU ≥ 0.5
- **FP** (false positive) — a box where there is no pedestrian. A **false alarm**.

And each real pedestrian you failed to find is:
- **FN** (false negative) — a **miss**.

### 2.2 The two numbers that matter here

Most of ML uses precision and recall. Pedestrian detection uses two closely-related numbers that
are more natural for a *safety* system:

**Miss rate** — of all the real pedestrians, what fraction did I fail to detect?

```
MR = FN / (number of real pedestrians) = 1 − recall
```
Lower is better. `MR = 0.05` means you missed 5% of the people.

**FPPI — False Positives Per Image** — how many false alarms do I raise, per image, on average?

```
FPPI = (total false positives) / (number of images)
```
Lower is better. `FPPI = 0.1` means one false alarm every ten images.

**Why FPPI and not precision?** Because it's *operationally meaningful*. If you're building a
system that brakes a car or alerts a drone operator, "one false alarm every 10 frames" is a
sentence a human can act on. Precision ("83% of my alerts are real") is much harder to reason
about at deployment. This subfield chose the metric that a *systems engineer* can use. That is a
good instinct and worth internalising.

### 2.3 One number isn't enough — you get a curve

Remember: the detector isn't one behaviour, it's a curve. Sweep the confidence threshold from
1.0 down to 0.0 and record (FPPI, MR) at each step:

| Threshold | Behaviour | FPPI | MR |
|---|---|---:|---:|
| 0.95 | Only reports what it's sure of | 0.01 | 0.60 |
| 0.70 | | 0.10 | 0.25 |
| 0.40 | | 0.50 | 0.12 |
| 0.10 | Reports anything that twitches | 1.00 | 0.08 |

Read it: as you allow more false alarms (FPPI ↑), you miss fewer people (MR ↓). **Always.** That's
the trade-off, made visible. Plot MR against FPPI (log-log axes) and you get the detector's
signature curve. A better detector's curve sits **lower** — fewer misses at the same false-alarm
budget.

### 2.4 MR⁻² — collapsing the curve into one number

We want a single number to rank methods. KAIST's answer:

> **Sample the curve at 9 FPPI points, log-spaced between 10⁻² and 10⁰ (i.e. 0.01 → 1.0), take the
> miss rate at each, and average them (geometric mean). That's the log-average miss rate, written
> MR⁻². Lower is better.**

The nine points are `0.01, 0.018, 0.032, 0.056, 0.10, 0.18, 0.32, 0.56, 1.0`.

The "⁻²" in the name just refers to that 10⁻² lower bound. It is **not** an exponent — don't let
the notation scare you.

**Make it concrete.** The KAIST test set is **2,252 images**. So:
- `FPPI = 1.0` → the detector is allowed ~2,252 false alarms in total.
- `FPPI = 0.01` → the detector is allowed only ~**22** false alarms across the entire test set.

That's the punchline: **MR⁻² deliberately measures you in the low-false-alarm regime** — the
regime you'd actually deploy in. It refuses to give you credit for performance you only achieve by
spraying thousands of false boxes.

So when `LITERATURE.md` says **TFDet gets 4.47**, it means: averaged over those realistic
operating points, it misses about **4.5%** of the pedestrians it's supposed to find.

### 2.5 Why NOT mAP — and why this matters for you personally

**mAP** (mean Average Precision) is the metric you already know from COCO. It averages precision
over *all* recall levels — including absurd operating points nobody would ever deploy.

Two consequences:

1. **Higher is better for mAP; LOWER is better for MR.** These run in *opposite directions*. You
   will misread a table at some point. Everyone does. Write `↓` next to every MR column.
2. **Reporting mAP alone makes your work incomparable to this entire literature.** Every number in
   `LITERATURE.md`'s big table is MR⁻². If you report mAP, you cannot place yourself in that table
   at all, and a reviewer can't check you.

And here's the part that should give you a small chill, given the APEX history: **mAP is exactly
the metric that hid your broken fusion.** A clean daytime mAP of 0.99 told you everything was
fine, while thermal was contributing literally nothing. The metric you choose determines the
failures you're *able to see.*

### 2.6 "Reasonable", and Day/Night

You'll see **"Reasonable subset"** everywhere. It means: only evaluate on pedestrians **≥ 55 px
tall** and **not heavily occluded**. Tiny, mostly-hidden people are excluded from the headline
number.

Is that cheating? A bit — and the literature is honest about it (`LITERATURE.md` §7.10 notes that
"All"-condition miss rates are still ~20%+, while Reasonable is down at 4.5%). But it's the shared
convention, so we follow it *and* we say so.

**Day / Night / All are reported as three separate columns, always.** This will turn out to be
where the interesting physics lives — see Part 5. Report only the aggregate and you will hide your
own most interesting result.

---

## Part 3 — ⭐ THE CENTRAL IDEA: feature maps, stride, and why "where you fuse" is the whole game

**This is the part to actually understand. Everything else in PRISM orbits it.**

### 3.1 What a backbone produces

Push an image through a CNN backbone (YOLOX's is called CSPDarknet). It doesn't produce "an
answer" — it produces a stack of **feature maps** at progressively lower resolution and higher
abstraction. For our input size of **640×512**:

| Stage | Name in code | Stride | Spatial size | Channels | What lives here |
|---|---|---:|---|---:|---|
| early | `dark3` | **8** | 80 × 64 | 128 | edges, textures, small blobs — *fine detail* |
| middle | `dark4` | **16** | 40 × 32 | 256 | parts — limbs, torsos |
| deep | `dark5` | **32** | 20 × 16 | 512 | "there is a person-ish thing here" — *semantics* |

**Stride** is the compression factor: how many input pixels one feature-map cell corresponds to.
Stride 8 means each cell of that 80×64 grid "looks at" an 8×8 patch of the original image.

Sanity-check it: 640/8 = 80, 512/8 = 64. ✓  640/32 = 20, 512/32 = 16. ✓

The pattern, and it's a genuine trade-off with no free lunch: **as you go deeper, you gain meaning
and lose location.** Early layers know precisely *where* an edge is but not what it belongs to.
Deep layers know there's a person but only roughly where.

### 3.2 Now the consequence that makes this project exist

Suppose the thermal image is **misaligned** by **8 pixels** relative to the RGB image. (Real
number: this is roughly the scale of misalignment in real dual-camera rigs, and the *entire*
problem with your Apollyon rig.)

What does an 8-pixel input shift *look like* to each stage of the network?

| Fuse at | Stride | 8 px input shift becomes… | Verdict |
|---|---:|---|---|
| `dark3` | 8 | **1.0 cell** — a whole cell off | Catastrophic. You're adding a person's features to the background beside them. |
| `dark4` | 16 | **0.5 cell** | Bad. Blurred and smeared. |
| `dark5` | 32 | **0.25 cell** — *sub-cell* | Barely noticeable. The features overlap anyway. |

**Read that table again. That is the thesis of the entire project.**

Fusing **deep** (high stride) is *automatically robust to misalignment*, because the spatial
pooling has already washed the shift away. Fusing **shallow** (low stride) demands near-perfect
pixel correspondence — and punishes you brutally when you don't have it.

This is why "middle fusion" is a real idea and not just a compromise:
- **Too early** (at the raw pixels) → one filter bank must serve two totally different image
  types, *and* you have zero misalignment tolerance.
- **Too late** (merge only the final boxes) → the streams never help each other see. A pedestrian
  that is faint in RGB and faint in thermal, but obvious when you combine the evidence, is lost
  forever — neither stream detected it alone, so there's nothing to merge.
- **The middle** → each stream keeps its own low-level filters (good), you merge at a semantic
  level where features mean the same thing in both modalities (good), *and* you inherit spatial
  tolerance to misalignment for free (great).

**And this is precisely why our headline experiment is the misalignment sweep.** On perfectly
aligned data, all these fusion choices look nearly identical (Part 6 — that's the null result we
expect). Break the alignment, and they fan out dramatically. The differences between operators
only *become visible* under stress. That's not a trick; that's how you find out what a design
choice actually buys you.

---

## Part 4 — What "fusion" concretely means

Strip away the words. You have two tensors at the same stage of two backbones:

```
F_rgb  : (B, 256, 40, 32)      ← from the RGB stream's dark4
F_therm: (B, 256, 40, 32)      ← from the thermal stream's dark4
```

You must produce **one** tensor of shape `(B, 256, 40, 32)` to hand onward to the neck. The
**fusion operator** is just: *how?* That's all it is.

The menu, cheapest first (this is `LITERATURE.md` §3, decoded):

| Operator | What it literally does | Params | Intuition |
|---|---|---:|---|
| **Sum** | `F_rgb + F_therm` | 0 | Trust both equally, always. Zero cost. |
| **Max** | `torch.maximum(...)` | 0 | Per-cell OR-gate: "whichever stream is more excited wins." Survives one stream dying. |
| **Product** | `F_rgb * F_therm` | 0 | AND-gate. **Dangerous** — kills anything only one stream sees, i.e. exactly the night-time pedestrian RGB can't see. |
| **Concat + 1×1 conv** | stack to 512 ch, conv down to 256 | 2C² ≈ 131k | **The workhorse.** Lets the network *learn* the mixing weights per channel. This is what you must beat. |
| **Learned scalar** | `α·F_rgb + β·F_therm` | 2 | Tests whether one global knob is all you need. |
| **SE / channel gating** | a tiny subnet looks at the features and reweights channels | ~2C²/r | Input-dependent: "it's dark — lean on thermal." |
| **Illumination gate** | a subnet classifies day/night and weights the streams | small | The explicit version of the above. |
| **Cross-attention** | every position attends to every position | large | Most expressive, most expensive. Can in principle *find* a shifted match — which is why it should win under misalignment. |

Note what's happening here: this list is ordered by *how much the operator is allowed to adapt to
the input*. Sum adapts not at all. Cross-attention adapts completely. The research question is:
**how much adaptivity do you actually need, and when?**

**Axis A** of our grid = *which stage* you do this at (`dark3` / `dark4` / `dark5` / all three).
**Axis B** = *which operator* from the table above.
That 2-D sweep, plus misalignment as a third axis, is PRISM.

---

## Part 5 — Why two cameras at all? (the physics, briefly, because it explains the results)

A thermal (LWIR, 8–14 µm) camera does not see light. It sees **emitted heat**. A human body is a
~310 K emitter and, crucially, human skin has emissivity ≈ 0.98 — nearly a perfect blackbody. **A
person glows.** That's the entire physical basis of this field.

Consequences you will literally see in your results table:

- **At night, thermal massively beats RGB.** RGB has no photons to work with; thermal doesn't care
  — the person is still hot. In the literature: RGB-only scores **27.42** MR at night, thermal-only
  **7.76**. That's not a small gap, it's a different universe. *Lower is better — thermal is ~3.5×
  better.*
- **In daylight, RGB beats thermal** (RGB 14.56 vs thermal 24.59). Colour, texture and context are
  genuinely informative, and thermal contrast is poor when everything is warm.
- **Thermal crossover:** on a hot day, when ambient temperature ≈ skin temperature, thermal
  contrast collapses toward zero and people *vanish* from thermal. This is why your **Day** column
  will be your problem column, not Night.
- **Glass is opaque to LWIR.** You cannot see a person through a windscreen or a window in thermal
  — you see the glass's own surface temperature. **This is the single best answer to "why not just
  use thermal and skip RGB?"** There are things RGB can see that thermal *physically cannot*. Not
  "does worse on" — *cannot*.

So the two modalities are genuinely complementary, in a way you can argue from physics rather than
from benchmark numbers. That argument belongs in your report's motivation section.

**And now the trap.** Look again: **thermal-only scores 18.99 overall; RGB-only 18.67.** They're
comparable overall, and thermal crushes RGB at night. So if your fancy fusion model scores 18.5,
you have **not** built a working fusion — you've built something barely better than one camera and
a piece of tape over the other. *This is the APEX failure, exactly.* Which is why the rule is:
**fusion must beat the better single modality, on all three columns, by more than the noise.**
Anything less is a null result, and we report it as one.

---

## Part 6 — The finding that should make you uncomfortable

From the real leaderboard in `LITERATURE.md`:

- The **entire spread** across a **100× range of fusion cost** — from free element-wise sum up to a
  259-GFLOP cross-attention transformer — is about **2 MR points**.
- **Cleaning up the annotations** was worth **~4 MR points** (11.63 → 7.45).
- Plain **concat** with good augmentation (MLPD, **7.58**) *beats* a clever attention mechanism
  (MBNet, **8.13**).

Sit with that. **On aligned data, the fusion architecture — the thing you were assigned to
research — is roughly half as important as fixing your labels.**

That is not a reason to give up. It is the most valuable thing in this document, and it is almost
certainly what Soham wants you to discover, because it generalises far past this project:

> **Most of what looks like architectural progress in ML is data quality, training protocol, and
> noise — wearing an architecture costume.**

It's why PRISM is built the way it is:
1. We **lock the data first** (Phase 0), because that's the biggest lever and it's not glamorous.
2. We **measure the seed noise floor** (Phase 1) before believing any result — if re-running the
   *same config* with a different random seed moves MR by ±0.8, then a 0.5-point "win" is
   **weather**, not a finding. The published literature quotes deltas of 0.3–1.0 with **no error
   bars anywhere**. Running 3 seeds makes you more rigorous than the papers you're citing.
3. We run the flat operator sweep anyway — **as a deliberate null control**, to prove we can
   measure honestly.
4. Then we go where the signal actually is: **misalignment** (Part 3).

---

## Part 7 — Reading `LITERATURE.md`'s big table

Now go back to §2.1 and it should decode. Here's the key with a couple of rows:

| Year | Method | Fusion type | MR All ↓ | Day ↓ | Night ↓ |
|---|---|---|---|---|---|
| 2015 | ACF+T+THOG | early | ≈47.3 | 42.6 | 56.2 |
| 2016 | Halfway Fusion | mid: concat @C4 | 25.75 | 24.88 | 26.59 |
| 2021 | MLPD | mid: **plain concat** + good training | 7.58 | 7.95 | 6.95 |
| 2022 | ProbEn3 | **late** (just merges boxes!) | 5.14 | 6.04 | 3.59 |
| 2025 | TFDet | mid: heavy machinery | **4.47** | 5.22 | 3.36 |

Things to notice, which are the actual lessons:
- **Everything is falling over time** (47 → 4.5). Lower is better. Progress is real.
- **The 2016 "Halfway Fusion" paper is literally your assignment, in its 2016 form.** They compared
  early / halfway / late fusion and found halfway won. You are redoing it with ten years of
  hindsight, a modern base, and — this is your addition — *under misalignment*.
- **ProbEn (5.14) is late fusion and it beat almost every clever mid-fusion method.** It just runs
  two detectors and merges their outputs probabilistically. This is the most humbling row in the
  table and it is why ProbEn is one of our mandatory controls: if we don't beat *it*, middle fusion
  has not justified itself.
- **Night is now better than Day** for the top methods (3.36 vs 5.22). Thermal has solved night.
  The remaining hard problem is daytime.

---

## Part 8 — Vocabulary you can now look up without fear

| Term | Plain meaning |
|---|---|
| **Backbone** | The CNN that turns an image into feature maps. (CSPDarknet, ResNet, VGG.) |
| **Neck** (FPN/PAN) | Mixes feature maps across stages so small and large objects are both handled. |
| **Head** | The final layers that emit boxes + classes. |
| **Stride** | Input pixels per feature-map cell. 8 / 16 / 32. **The most important number in Part 3.** |
| **Anchor-free** | Predict boxes directly, no template boxes. YOLOX is this. |
| **SimOTA** | YOLOX's rule for deciding *which* grid cells are responsible for *which* ground-truth object during training. ("Label assignment.") |
| **NMS** | Deletes duplicate overlapping boxes. |
| **IoU** | Box overlap ratio. ≥0.5 = a hit. |
| **MR⁻²** | Log-average miss rate. **Lower is better.** The metric. |
| **FPPI** | False alarms per image. The x-axis of the curve. |
| **Reasonable** | Only pedestrians ≥55 px, not heavily occluded. The headline subset. |
| **LWIR** | Long-wave infrared, 8–14 µm. What "thermal" means here. |
| **Early / middle / late fusion** | Merge at pixels / at features / at output boxes. |
| **Ablation** | Change exactly ONE thing, re-measure, attribute the difference to that one thing. |
| **Seed noise floor** | How much your score moves when you change *nothing but* the random seed. Your bullshit detector. |

---

## What to do now

1. **Re-read this primer's Part 3 until the stride table feels obvious.** It is the load-bearing
   idea. If an 8-pixel shift being "1 cell at stride 8 but 0.25 cells at stride 32" feels natural,
   you have the project.
2. **Then** re-read `LITERATURE.md` §1 (the benchmark + the annotations) and §2.1 (the table). It
   should now read as *information* rather than noise.
3. **Then** the YOLOX paper, with one question in mind: *where in this architecture would I attach
   a second stream, and at what stride?*

And bring me the parts that still don't land. Confusion is information — it tells us which chapter
of the textbook to write next.
