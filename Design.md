# Design log

Decisions made while porting the research code (`Gaze_project`, private) into this
release repository. Newest entries last. Paths under "Baseline" refer to the private
research repository.

## 2026-10-11 — Port the two-party inference pipeline

**Question / goal:** give the release a runnable version of the pipeline that turns
two speech tracks, a scene description and a tagged script into gaze points and
head angles, without the hardcoded paths and notebook state of the research code.

**Baseline:** `gaze_framework_dynamic_scene.ipynb` and the modules it actually calls:
`prototypes/Gaze_aversion_prior/Evan_model.py` (classifier wrapper),
`prototypes/InputDataStructures.py` (`AgentInfo_final` only),
`prototypes/MVP/MVP_gaze_path_planner.py` (`Responsive_planner_Differnet_Targets` only),
`prototypes/EyeCatch/Saccade_model_with_internal_model.py`
(`InternalModelCenterBias_dynamic_scene`, `SacccadeGenerator_dynamic_scene`),
`prototypes/Optimization_based_head_eye_seperator/Baseline_optimization.py`
(`optimize_for_head_gaze_breakdown_dynamic_scene`),
`prototypes/Jin2019/EyeHeadDecomposition.py` (the two `Heuristic_decomposition_*`
classes), and the helper functions these import from `EvansToolBox/Utils` (MIT).
Not reused: every other class in those files (older generations, three-party and
listener variants), the GMM decomposition and its 656 model files, `JaliNeck`.

**Decision:** copy the logic of those classes and notebook cells line for line into
the `s3gaze` package. Only imports, paths, file layout and a few identifiers with
typos change. Numerical behaviour is preserved, including behaviour that looks
unintended (listed under "Known quirks" in the README), because the shipped
checkpoint and the paper results were produced with it.

**Basis:** explicit user decision (release the code the paper used); the list of
classes on the live path is a measured fact from reading the notebook's calls.

**Alternatives rejected:** (1) publishing the notebooks as they are — they need two
private repositories on `sys.path` and four machine-specific input cells; (2) fixing
the quirks during the port — that would change results with no way to attribute the
difference.

**Risk / what to watch:** a transcription slip during the port. Checked on one
synthetic two-speaker clip (a real speech file split into alternating 4 s turns, with
made-up word timings): classifier input features, classifier logits, planner target
sequence and the eye and head curves of the port are identical (difference 0.0) to
the research classes given the same input and seeds. Not checked: a real clip from
the paper, Whisper transcription, and the tag functions against the notebook (they
are the same extracted code, but were not compared side by side).

## 2026-10-11 — Remove the JALI dependency

**Question / goal:** the pipeline must run without any JALI Research software or
files.

**Baseline:** in the research notebook JALI appears in three places: (1)
`NeckCurve` from `prototypes/JaliNeck/JaliNeck.py`, a speech-driven neck model that
reads a JALI phoneme alignment (`*_PraatOutput.txt`); (2) a
`Sentence_word_phone_parser(praatoutput_path, ...)` call in the notebook whose result
is never passed to the planner; (3) `Aversion111Prior.predict` writing plain-text
scripts (`<name>_<speaker>.txt`, `..._tagged.txt`) for JALI's lip-sync.

**Decision:** none of the three is ported. Each removal is marked in the code with a
`JALI:` comment saying what used to happen there. The `other_neck` slot in the output
keeps its key and is always empty.

**Basis:** explicit user decision ("remove the jali dependency and annotate it").

**Alternatives rejected:** replacing the phoneme alignment with an open forced
aligner so the neck model could stay — the neck model itself is not ours to release,
and writing a replacement would be new, unvalidated science.

**Risk / what to watch:** head motion is less lively than in the paper videos, which
included the speech-driven neck layer. Head roll is zero unless a neck curve is
supplied (next entry).

## 2026-10-11 — Neck motion is an optional input

**Question / goal:** the release has no neck model. Say so clearly and still let a
user who has neck curves from elsewhere use them.

**Baseline:** the notebook's "Compute Neck Motion (from CSV)" cell (reads
`<name>_<speaker>_neck.csv`, three columns at 20 fps, scales by a factor, one pass of
Laplacian smoothing) and `maya_scripts/load_gaze_and_neck_animation_with_ambience.py`,
which adds the neck curves to the gaze-driven head rotation on the rig.

**Decision:** keep the CSV reader as an optional input. When the file is absent the
neck layer is empty. For the plain (non-Maya) export, the neck curve is added to the
gaze-driven head angles with the same signs the Maya loader uses:
`yaw = azimuth - neck_y`, `pitch = elevation + neck_x`, `roll = -neck_z`, with the
neck curve linearly interpolated onto the output time line.

**Basis:** explicit user decision ("we don't have the neck model … highlight that it
is an optional input"); signs read from the Maya loader.

**Alternatives rejected:** dropping neck input entirely — the output format and the
Maya loader already carry the slot, and removing it would break existing pickles.

**Risk / what to watch:** the sign convention is the JALI rig's. A neck curve from a
different source may need its axes flipped. Maya interpolates keys with its own
tangents; the plain export uses linear interpolation, so the two can differ slightly
between keys.

## 2026-10-11 — Port the preprocessing pipeline as scripts

**Question / goal:** release the code that turns per-frame gaze and head estimates
into aversion labels and training features, and make it the headline of the README.

**Baseline:** `Dataset_Util/dataset_deep_learning_preprocess.ipynb`, cells 1, 7, 11,
14-15, 17-18, 24, 30, 35-36, 38, 45-46. This is the path that produced the features
the shipped checkpoint was trained on (65 audio + 6 word-timing + 6 sentence-timing
values per speaker, fixed 10 s windows).

**Decision:** one function per stage in `s3gaze/preprocessing/stages.py`, same order as
the notebook, run through `scripts/preprocess.py --stages`. Not ported: the BERT
text features (cells 26-27, used only by an early baseline), the windowing of
direction labels (cells 48-49, unused by the shipped model), and the later
"realtime augmentation" notebook (POS tags, WavLM features), which belongs to model
variants that need ground-truth labels at inference.

**Basis:** upstream code; the choice of path follows from which checkpoint ships
(`training/training.ipynb`, April 2023, per the checkpoint's internal name and
config).

**Alternatives rejected:** porting the later preprocessing path as well — it serves
models that are not being released in this pass.

**Risk / what to watch:** two steps are ported exactly as written although they look
unintended, because they shaped the training data: the fixation fill loop (gaps
between fixations stay at gaze (0, 0)) and the sentence-timing window resampling
(source time axis starts at the window's frame index instead of 0). Both are marked
`NOTE(port)` in the code and listed in the README. The source dataset is not on this
machine, so stages 1-3 were not run at all and stages 4-5 were only run on one
synthetic shot to check that they execute and produce the expected shapes.
