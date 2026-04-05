# FYP Final Report Master Outline

Last updated: 2026-04-02  
Role of this file: central writing-control document for the entire FYP final report.  
Usage rule: every later chapter draft should be checked against this file first, and this file should be updated whenever chapter scope, contribution framing, evidence mapping, or reproducibility requirements change.

Additional note:
- the earlier standalone metrics review file has been consolidated into [experiment_execution_and_metrics_plan.md](/Users/wuyuchen/Desktop/FYP_final/notes/experiment_execution_and_metrics_plan.md), which now serves as the combined note for experiment rerun planning, metric selection, and figure/table display policy.

---

# 1. Report Title and Project Positioning

## Working Title
**Route-Aware Multi-Branch Deepfake Detection with Hybrid Manifold Routing and Region-Level Patch/Pair Evidence**

Alternative shorter title:
**Deepfake Detection via Hybrid Routing and Route-Aware Fusion of Region-Level Patch and Pair Cues**

## One-Sentence Project Goal
To build a deepfake detection pipeline that is not only accurate on known manipulation groups but also more robust under cross-method and cross-domain evaluation.

## One-Sentence Method Summary
The final mainline combines a frozen visual backbone for CLS and patch-token extraction, facial-region aggregation via face parsing, a hybrid manifold upstream route model for coarse fake-group routing, and a downstream route-aware fusion head that integrates patch and pair evidence for final real/fake prediction.

## One-Sentence Core Contribution
The project contributes a full end-to-end structured detection pipeline that couples geometry-aware routing with region-level downstream fusion, together with a cache-based reproducible experimental framework and a large body of ablation/failed-attempt evidence showing systematic validation rather than one-off implementation.

## Positioning Statement for the Report
This report should position the project as:
- a **research-style detection system** rather than only an engineering demo;
- a **method-and-analysis driven FYP**, not merely “training another classifier”;
- a project that includes both **algorithmic design** and **system setup / reproducibility discipline**;
- a project with a **clear final mainline** and a substantial set of **ablation and discarded alternatives** that demonstrate iterative research thinking.

## Mainline Thesis to Keep Consistent Across the Whole Report
The report should consistently present the final system as:
- Upstream: hybrid manifold route model on CLS embeddings.
- Mid-layer engineering: face-region parsing and compact region cache construction.
- Downstream: patch branch + pair branch + route-aware meta fusion.
- Final decision: thresholded fake probability calibrated on validation.

Do not let later chapters drift into describing the project as:
- just a patch classifier;
- just a routing classifier;
- just a generic fusion experiment;
- or a loose collection of ablations without a coherent final pipeline.

---

# 2. Proposed Final Report Structure

The structure below is designed to satisfy both a research-paper style narrative and the handbook requirement that system setup / reproducibility must appear explicitly.

## Front Matter
- Title page
- Declaration / acknowledgements if required by school format
- Abstract
- Table of contents
- List of figures
- List of tables
- List of abbreviations / notation if needed

## Abstract
- Problem context
- Main method
- Core implementation structure
- Main experimental finding
- One sentence on contribution / significance

## Chapter 1 Introduction
### 1.1 Background and Motivation
### 1.2 Problem Statement
### 1.3 Why This Problem Is Difficult
### 1.4 Project Objectives
### 1.5 Overview of Proposed Solution
### 1.6 Main Contributions
### 1.7 Report Structure

## Chapter 2 Literature Review
### 2.1 Deepfake Detection Landscape
### 2.2 Visual Representation Approaches
### 2.3 Region-Aware / Local Artifact Modelling
### 2.4 Metric Learning / Prototype / Geometry-Oriented Approaches
### 2.5 Multi-Branch Fusion and Routing Ideas
### 2.6 Domain Shift and Cross-Manipulation Generalisation
### 2.7 Gap Analysis and Positioning of This Project

## Chapter 3 Background and Problem Formulation
### 3.1 Task Definition
### 3.2 Dataset Splits and Evaluation Setting
### 3.3 Label Space and Fake Group Structure
### 3.4 Training and Inference Objectives
### 3.5 Notation and Mathematical Setup
### 3.6 Design Requirements Derived from the Problem

## Chapter 4 Proposed Method
### 4.1 System Overview
### 4.2 Frozen Backbone Feature Extraction
### 4.3 Facial-Region Patch Aggregation
### 4.4 Upstream Hybrid Manifold Route Model
### 4.5 Patch Branch
### 4.6 Pair Branch
### 4.7 Route-Aware Meta Head
### 4.8 Inference Pipeline and Threshold Decision
### 4.9 no-FR Variant and Design Rationale

## Chapter 5 System Implementation and Reproducibility
### 5.1 Repository Structure and Mainline vs Backup
### 5.2 Runtime Pipeline and Script Entry Points
### 5.3 Cache Layers: CLS / Patch / Compact Patch
### 5.4 Checkpoints, Serialized Branch Bundles and Replay
### 5.5 Environment, Dependencies and Hardware
### 5.6 Cluster Scripts / Batch Execution
### 5.7 Reproduction Procedure
### 5.8 What Is Included in Main Text vs Appendix

## Chapter 6 Experimental Setup
### 6.1 Experimental Questions
### 6.2 Data Sources and Split Definitions
### 6.3 Training Configuration
### 6.4 Validation and Threshold Selection
### 6.5 Evaluation Metrics
### 6.6 Mainline Variants Compared
### 6.7 Notes on Fairness / Constraints / Practical Choices

## Chapter 7 Experimental Results
### 7.1 Mainline Performance on Test-FF
### 7.2 Mainline Performance on OOD / Cross-Domain Evaluation
### 7.3 Route-Level Behaviour
### 7.4 Branch-Level Behaviour
### 7.5 Final Chosen Variant
### 7.6 Result Summary

## Chapter 8 Ablation Studies and Analysis
### 8.1 Fusion Family Evolution
### 8.2 Pair-Region Subset Ablation
Additional refinement note:
- the draft now explicitly records the likely lineage of the earlier `canonical` subset: an earlier pair-first region-ranking script appears to have informed the later hard-coded canonical regions;
- `FS`, `FR`, and `FE` are framed as pair-conditioned canonical subsets because they belong to the paired fake-real groups in the early pipeline;
- `EFS` is now framed as a ranking-supported but structurally distinct case, because it lacks the same paired real-reference supervision and is better explained as a stable top-three region retention from early ranking outputs.
### 8.3 Threshold Sweep and Calibration Trade-Off
### 8.4 no-FR Ablation
### 8.5 Route-Conditioned Rule / Strategy Search
### 8.6 Alternative Heads and Failed Replacements
### 8.7 What Was Revalidated Repeatedly

## Chapter 9 Discussion, Contributions and Conclusion
### 9.1 Interpretation of Main Findings
### 9.2 Student's Own Technical Contributions
### 9.3 Engineering Contributions
### 9.4 Limitations
### 9.5 Future Work
### 9.6 Final Conclusion

## References

## Appendices
### Appendix A Mathematical Details
### Appendix B Additional Experimental Tables
### Appendix C Additional Reproduction Commands / Config Snapshots
### Appendix D Extra Ablation Notes or Per-Method Breakdowns
### Appendix E Supplementary Repository / Output Directory Notes

---

# 3. Section-by-Section Writing Plan

Status vocabulary:
- `not started`
- `outline done`
- `draft done`
- `revised`
- `finalized`

## Abstract
- Purpose:
  Provide a compact research-style summary of problem, method, implementation scope, and findings.
- Must include:
  Deepfake detection context; route-aware multi-branch method; mention hybrid manifold routing and patch/pair/meta fusion; one main result trend; one contribution sentence.
- Should avoid:
  Long background; too many numbers; file-level engineering detail; excessive ablation content.
- Evidence sources:
  Mainline code inventory; formula notes; final experiment summary; replay outputs.
- Status:
  `draft done`
- Abstract draft now covers:
  project objective; the retained route-aware multi-branch method; the cache-and-replay implementation scope; the main result trend across friendlier and harder evaluation settings; and the overall conclusion and contribution framing.
- Remaining Abstract needs:
  - align the final wording with the definitive Chapter 7 quantitative presentation once the main result tables are fixed
  - optionally add one compact citation-free phrase adjustment after the title and contribution wording are finalized across Chapters 1 and 9

## Chapter 1 Introduction

### 1.1 Background and Motivation
- Purpose:
  Explain why deepfake detection matters and why robustness/generalisation matters.
- Must include:
  Manipulation diversity; domain shift; why image-level real/fake is insufficient if model overfits artifacts.
- Should avoid:
  Detailed literature taxonomy; implementation details.
- Evidence sources:
  Literature review notes; project motivation notes; cross-domain result trends.
- Status:
  `draft done`

### 1.2 Problem Statement
- Purpose:
  Define the project objective precisely.
- Must include:
  Detect real vs fake; use fake-group structure during modelling; explain train/test_ff/ood setting.
- Should avoid:
  Jumping straight into method before defining task.
- Evidence sources:
  Current code pipeline; split handling in `train_upstream.py` and `train_downstream_head.py`.
- Status:
  `draft done`

### 1.3 Why This Problem Is Difficult
- Purpose:
  Make the methodological choices look necessary rather than arbitrary.
- Must include:
  intra-class variability; inter-method overlap; real-side failure under OOD; local artifact heterogeneity.
- Should avoid:
  Unsubstantiated claims beyond project evidence.
- Evidence sources:
  ablation reports; threshold sweeps; route-conditioned analysis.
- Status:
  `draft done`

### 1.4 Project Objectives
- Purpose:
  Convert motivation into concrete technical targets.
- Must include:
  build structured pipeline; exploit fake-group routing; incorporate local region evidence; support reproducible experiments.
- Should avoid:
  vague objectives like “improve AI”.
- Evidence sources:
  final pipeline summary; code organization.
- Status:
  `draft done`

### 1.5 Overview of Proposed Solution
- Purpose:
  Give the first short pipeline description.
- Must include:
  frozen backbone -> route -> patch/pair -> route-aware meta head -> thresholded decision.
- Should avoid:
  full mathematical derivation.
- Evidence sources:
  `main.py`, `train_upstream.py`, `train_downstream_head.py`.
- Status:
  `draft done`

### 1.6 Main Contributions
- Purpose:
  State contributions in report language.
- Must include:
  one method contribution; one engineering/reproducibility contribution; one experimental/analysis contribution.
- Should avoid:
  overstating failed attempts as final contributions.
- Evidence sources:
  contribution tracker in this file.
- Status:
  `draft done`

### 1.7 Report Structure
- Purpose:
  Guide the reader.
- Must include:
  one paragraph overview of chapters.
- Should avoid:
  too much detail.
- Evidence sources:
  this master outline.
- Status:
  `draft done`

- Overall Chapter 1 status:
  `draft done`
- Chapter 1 evidence package:
  [notes/student_contributions.md](/Users/wuyuchen/Desktop/FYP_final/notes/student_contributions.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md); [chapters/ch7_results.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch7_results.md); [chapters/ch8_ablation_analysis.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch8_ablation_analysis.md); [notes/references_todo.md](/Users/wuyuchen/Desktop/FYP_final/notes/references_todo.md); [chapters/ch1_introduction.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch1_introduction.md).
- Chapter 1 draft now covers:
  background and motivation; problem statement; significance and key challenges; project objectives; high-level solution overview; explicit student contribution framing; and report organization in a research-style introduction.
- Remaining Chapter 1 needs:
  - add literature-backed motivation references once Chapter 2 citation choices are fixed
  - decide whether `no-FR` is named explicitly in the Introduction contribution list or deferred to later chapters
  - align final significance and contribution wording with the eventual Chapter 9 discussion/conclusion phrasing

## Chapter 2 Literature Review

### 2.1-2.6 Review Sections
- Purpose:
  Show understanding of prior work and position this project correctly.
- Must include:
  detection methods; representation learning; local artifact modelling; routing/fusion ideas; domain generalisation.
- Should avoid:
  discussing the current codebase here in implementation terms.
- Evidence sources:
  [notes/references_todo.md](/Users/wuyuchen/Desktop/FYP_final/notes/references_todo.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md); selected project-positioning cues from [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md).
- Status:
  `draft done`

### 2.7 Gap Analysis and Positioning
- Purpose:
  Bridge literature and this project.
- Must include:
  why hybrid routing + local evidence fusion is justified; why reproducibility matters; why cross-domain behaviour is important.
- Should avoid:
  turning into method section too early.
- Evidence sources:
  [notes/references_todo.md](/Users/wuyuchen/Desktop/FYP_final/notes/references_todo.md); literature summary to be collected; project pipeline summary from [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md) and [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md).
- Status:
  `draft done`

- Overall Chapter 2 status:
  `draft done`
- Chapter 2 evidence package:
  [notes/references_todo.md](/Users/wuyuchen/Desktop/FYP_final/notes/references_todo.md); [chapters/ch2_literature_review.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch2_literature_review.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md); [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md).
- Chapter 2 draft now covers:
  a focused literature review of the deepfake detection landscape; generalized and cross-manipulation detection; frozen or foundation-model-based visual representations; region-aware semantic local evidence; pairwise, prototype, manifold, and geometry-oriented ideas; multi-branch fusion and routing; and a final gap-analysis section that naturally motivates the proposed method without turning into a second method chapter.
- Remaining Chapter 2 needs:
  - replace placeholder citations such as `[Ref-LR-*]` with the final verified ACM author-year references
  - collect and verify the final paper list for each review strand, especially generalized detection, geometry-aware classification, and routing/fusion references
  - confirm the exact published source behind `DF40`, if one exists and should be cited in Chapter 2 or Chapter 3
  - decide how much face parsing provenance belongs in the literature review versus being deferred to implementation-oriented chapters
  - keep the revised draft selective so it supports Chapter 4 rather than expanding into a broad survey

## Chapter 3 Background and Problem Formulation

### 3.1 Task Definition
- Purpose:
  Define the supervised task formally.
- Must include:
  binary final decision plus auxiliary fake-group routing.
- Should avoid:
  implementation detail like filenames.
- Evidence sources:
  [notes/math_formulas.md](/Users/wuyuchen/Desktop/FYP_final/notes/math_formulas.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md); training-code-derived notation already consolidated in notes.
- Status:
  `outline done`

### 3.2 Dataset Splits and Evaluation Setting
- Purpose:
  Define train / test_ff / ood regime.
- Must include:
  what each split is used for; what kind of generalisation is being measured.
- Should avoid:
  exhaustive appendix-like directory dumps.
- Evidence sources:
  [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md); [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md); [chapters/ch6_experimental_setup.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch6_experimental_setup.md).
- Status:
  `draft done`

### 3.3 Label Space and Fake Group Structure
- Purpose:
  Explain EFS / FS / FR / FE / REAL and no-FR variant.
- Must include:
  full and no-FR label schemes; why these labels matter.
- Should avoid:
  hiding no-FR until ablation chapter.
- Evidence sources:
  [notes/math_formulas.md](/Users/wuyuchen/Desktop/FYP_final/notes/math_formulas.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md); [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md).
- Status:
  `draft done`

### 3.4 Training and Inference Objectives
- Purpose:
  Introduce the mathematical problem before chapter 4.
- Must include:
  route probability, branch scores, final threshold decision.
- Should avoid:
  reproducing every implementation detail twice.
- Evidence sources:
  [notes/math_formulas.md](/Users/wuyuchen/Desktop/FYP_final/notes/math_formulas.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md).
- Status:
  `draft done`

### 3.5 Notation and Mathematical Setup
- Purpose:
  Normalize symbols for later method chapter.
- Must include:
  CLS embedding, patch tokens, region features, route vector, branch outputs, threshold.
- Should avoid:
  code-variable-only presentation.
- Evidence sources:
  [notes/math_formulas.md](/Users/wuyuchen/Desktop/FYP_final/notes/math_formulas.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md).
- Status:
  `draft done`

### 3.6 Design Requirements Derived from the Problem
- Purpose:
  State what the method must achieve.
- Must include:
  coarse routing; region-level local evidence; efficient repeated experiments; robust real/fake calibration.
- Should avoid:
  claiming these were all solved perfectly.
- Evidence sources:
  [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md); [chapters/ch8_ablation_analysis.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch8_ablation_analysis.md).
- Status:
  `draft done`

- Overall Chapter 3 status:
  `draft done`
- Chapter 3 evidence package:
  [notes/math_formulas.md](/Users/wuyuchen/Desktop/FYP_final/notes/math_formulas.md); [chapters/ch3_background_problem.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch3_background_problem.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md); [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md); [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md); [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md).
- Chapter 3 draft now covers:
  formal task definition; the role of friendly versus shifted evaluation conditions; full and `no-FR` label-space formulations; CLS, patch-token, and region-level representation background; compact training and inference problem formulation; and design requirements that motivate the hierarchical method in Chapter 4.
- Remaining Chapter 3 needs:
  - decide the final boundary between Chapter 3 objective notation and Chapter 4 method equations so the two chapters do not duplicate each other
  - confirm how explicitly `DF40`, `test_ff`, and OOD split names should appear in Chapter 3 versus being deferred to Chapter 6
  - make one final pass to keep Chapter 3 at the problem-formulation level rather than letting it drift into implementation or ablation discussion
  - check final consistency of group notation and terminology against Chapters 4 and 6 once those chapters are revised together

## Chapter 4 Proposed Method

### 4.1 System Overview
- Purpose:
  Present the full architecture cleanly.
- Must include:
  one overall diagram; mainline-only path; clear separation between upstream and downstream.
- Should avoid:
  mixing in all ablation branches.
- Evidence sources:
  code inventory; pipeline map; `main.py`.
- Status:
  `outline done`

### 4.2 Frozen Backbone Feature Extraction
- Purpose:
  Explain input representation.
- Must include:
  CLS embedding and patch-token extraction; frozen backbone assumption.
- Should avoid:
  excessive low-level API detail.
- Evidence sources:
  `src/prepare/backbone.py`, `src/prepare/extractors.py`.
- Status:
  `draft done`

### 4.3 Facial-Region Patch Aggregation
- Purpose:
  Explain how patch tokens become region-level features.
- Must include:
  face parsing; merged regions; compact region cache; mean pooling.
- Should avoid:
  demo overlay implementation details.
- Evidence sources:
  `src/prepare/face_regions.py`, `src/prepare/cache.py`, math notes.
- Status:
  `draft done`

### 4.4 Upstream Hybrid Manifold Route Model
- Purpose:
  Explain the most methodologically distinctive upstream component.
- Must include:
  linear branch; manifold branch; centers/subspaces/offsets; pair prototypes; total objective.
- Should avoid:
  drowning the reader in training heuristics before introducing the main idea.
- Evidence sources:
  `src/train/train_upstream.py`; formula notes.
- Status:
  `draft done`

### 4.5 Patch Branch
- Purpose:
  Explain region-delta based patch evidence.
- Must include:
  region delta features; global patch classifier; group experts.
- Should avoid:
  overclaiming patch branch as the whole method.
- Evidence sources:
  `src/train/train_downstream_head.py`; formula notes.
- Status:
  `draft done`

### 4.6 Pair Branch
- Purpose:
  Explain paired-delta inspired group-wise evidence.
- Must include:
  paired real references; mean delta direction; cos+norm features; group-specific classifiers.
- Should avoid:
  treating pair branch as generic contrastive learning if the code does not do that.
- Evidence sources:
  `src/train/train_downstream_head.py`; pair-region reports; formula notes.
- Status:
  `draft done`

### 4.7 Route-Aware Meta Head
- Purpose:
  Present the final fusion mechanism.
- Must include:
  route-based weights; dynamic expert score; meta feature vector; route-aware experts; threshold decision.
- Should avoid:
  describing legacy fusion families here.
- Evidence sources:
  `src/train/train_downstream_head.py`, `main.py`, formula notes.
- Status:
  `draft done`

### 4.8 Inference Pipeline and Threshold Decision
- Purpose:
  Close the method chapter with the deployed inference path.
- Must include:
  end-to-end equation chain and binary decision rule.
- Should avoid:
  extended threshold sweep discussion.
- Evidence sources:
  `main.py`; formula notes.
- Status:
  `draft done`

### 4.9 no-FR Variant and Design Rationale
- Purpose:
  Introduce no-FR as a deliberate design variant.
- Must include:
  why no-FR was considered; what changes in labels/groups; relation to robustness.
- Should avoid:
  placing all no-FR results here.
- Evidence sources:
  current mainline code; no-FR reports in backup.
- Status:
  `draft done`

- Overall Chapter 4 status:
  `draft done`
- Chapter 4 evidence package:
  [notes/math_formulas.md](/Users/wuyuchen/Desktop/FYP_final/notes/math_formulas.md); [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md); [notes/student_contributions.md](/Users/wuyuchen/Desktop/FYP_final/notes/student_contributions.md); [main.py](/Users/wuyuchen/Desktop/FYP_final/main.py); [src/train/train_upstream.py](/Users/wuyuchen/Desktop/FYP_final/src/train/train_upstream.py); [src/train/train_downstream_head.py](/Users/wuyuchen/Desktop/FYP_final/src/train/train_downstream_head.py); [src/prepare/backbone.py](/Users/wuyuchen/Desktop/FYP_final/src/prepare/backbone.py); [src/prepare/cache.py](/Users/wuyuchen/Desktop/FYP_final/src/prepare/cache.py); [src/prepare/face_regions.py](/Users/wuyuchen/Desktop/FYP_final/src/prepare/face_regions.py); [notes/figures_tables_plan.md](/Users/wuyuchen/Desktop/FYP_final/notes/figures_tables_plan.md); [chapters/ch4_method.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch4_method.md).
- Chapter 4 draft now covers:
  system overview; frozen backbone feature extraction; facial-region aggregation; hybrid manifold routing; patch branch; pair branch; route-aware meta fusion; final threshold decision; no-FR design rationale.
- Remaining Chapter 4 needs:
  - create the Chapter 4 figures:
    overall pipeline, upstream routing, downstream route-aware fusion
  - decide whether the expanded upstream auxiliary-loss breakdown should stay in main text or be compressed into Appendix A
  - make one final pass on the boundary with Chapter 3 so notation/objective material is not repeated mechanically
  - lightly tighten wording once Chapter 3 is drafted, especially around problem formulation and no-FR motivation

## Chapter 5 System Implementation and Reproducibility

### 5.1 Repository Structure and Mainline vs Backup
- Purpose:
  Prevent confusion about what was actually used.
- Must include:
  `FYP_final` as clean runtime snapshot; `FYP_final_backup` as research archive; active vs legacy modules.
- Should avoid:
  pretending every file is part of the final pipeline.
- Evidence sources:
  repo inventory; README; backup survey.
- Status:
  `draft done`

### 5.2 Runtime Pipeline and Script Entry Points
- Purpose:
  Show how the system is executed in practice.
- Must include:
  `main.py`, `train_upstream.py`, `train_downstream_head.py`, prepare modules, replay/sample entry points.
- Should avoid:
  listing every helper script equally.
- Evidence sources:
  code inventory and pipeline map.
- Status:
  `draft done`

### 5.3 Cache Layers: CLS / Patch / Compact Patch
- Purpose:
  Show the engineering architecture that supports experimentation.
- Must include:
  what each cache stores; why compact patch cache matters.
- Should avoid:
  omitting the importance of caching for repeat experiments.
- Evidence sources:
  `src/prepare/cache.py`, `src/prepare/extractors.py`.
- Status:
  `draft done`

### 5.4 Checkpoints, Serialized Branch Bundles and Replay
- Purpose:
  Explain reproducible artifact flow.
- Must include:
  upstream checkpoint; patch/pair/head bundles; head meta; replay.
- Should avoid:
  loose claims like “the model can be rerun” without showing how.
- Evidence sources:
  `main.py`, `train_downstream_head.py`, outputs/replay.
- Status:
  `draft done`

### 5.5 Environment, Dependencies and Hardware
- Purpose:
  Satisfy handbook setup requirement.
- Must include:
  software stack; key libraries; Python/runtime environment; hardware/cluster/MPS/GPU notes if relevant.
- Should avoid:
  burying this in appendix only.
- Evidence sources:
  [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md); [environment.yml](/Users/wuyuchen/Desktop/FYP_final/environment.yml); [requirements.txt](/Users/wuyuchen/Desktop/FYP_final/requirements.txt); README; cluster scripts.
- Status:
  `draft done`

### 5.6 Cluster Scripts / Batch Execution
- Purpose:
  Demonstrate experiment organization at scale.
- Must include:
  cache generation and replay/training scripts on cluster; what they were used for.
- Should avoid:
  overdescribing obsolete cluster paths as current defaults.
- Evidence sources:
  [scripts/cluster](/Users/wuyuchen/Desktop/FYP_final/scripts/cluster); [FYP_final_backup/scripts/cluster](/Users/wuyuchen/Desktop/FYP_final_backup/scripts/cluster); [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md).
- Status:
  `draft done`

### 5.7 Reproduction Procedure
- Purpose:
  Provide a reader-followable path to rerun main experiments.
- Must include:
  required artifacts; cache dependencies; training order; replay order; known caveats.
- Should avoid:
  pretending exact reproduction is turnkey if local cache dependencies remain.
- Evidence sources:
  [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md); [pipeline_manifest.json](/Users/wuyuchen/Desktop/FYP_final/pipeline_manifest.json); [main.py](/Users/wuyuchen/Desktop/FYP_final/main.py); README; [outputs/replay](/Users/wuyuchen/Desktop/FYP_final/outputs/replay).
- Status:
  `draft done`

### 5.8 What Is Included in Main Text vs Appendix
- Purpose:
  Control report length.
- Must include:
  what setup details stay in main text; what command-level detail moves to appendix; how portability caveats should remain in the main chapter.
- Should avoid:
  dumping shell commands in the method chapter.
- Evidence sources:
  this master outline; [report_style_rules.md](/Users/wuyuchen/Desktop/FYP_final/report_style_rules.md); [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md).
- Status:
  `draft done`

- Overall Chapter 5 status:
  `draft done`
- Chapter 5 evidence package:
  [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md); [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md); [notes/student_contributions.md](/Users/wuyuchen/Desktop/FYP_final/notes/student_contributions.md); [README.md](/Users/wuyuchen/Desktop/FYP_final/README.md); [environment.yml](/Users/wuyuchen/Desktop/FYP_final/environment.yml); [requirements.txt](/Users/wuyuchen/Desktop/FYP_final/requirements.txt); [pipeline_manifest.json](/Users/wuyuchen/Desktop/FYP_final/pipeline_manifest.json); [main.py](/Users/wuyuchen/Desktop/FYP_final/main.py); [scripts/cluster](/Users/wuyuchen/Desktop/FYP_final/scripts/cluster); [notes/figures_tables_plan.md](/Users/wuyuchen/Desktop/FYP_final/notes/figures_tables_plan.md).
- Remaining Chapter 5 questions:
  - finalize path-neutral wording for the cache/history dependency caveat
  - decide whether the main text should include one concise environment table
  - curate a short appendix command set without turning Chapter 5 into a manual
- Chapter 5 draft coverage now includes:
  repository organization; runtime execution path; cache architecture; serialized artifact chain and replay; environment/hardware assumptions; cluster support role; high-level reproduction workflow; portability and cache-dependency caveats.
- Chapter 5 still needs refinement on:
  - possible inclusion of one concise environment table
  - final wording of the portability caveat to avoid over-specific local path emphasis
  - cross-checking Appendix C command material so the main chapter stays concise

## Chapter 6 Experimental Setup

### 6.1 Experimental Questions
- Purpose:
  Organize experiments around questions rather than around files.
- Must include:
  mainline effectiveness; robustness under OOD; contribution of patch/pair/meta; effect of no-FR; threshold trade-off.
- Should avoid:
  random chronology of script execution.
- Evidence sources:
  repeated findings; ablation reports.
- Status:
  `not started`

### 6.2 Data Sources and Split Definitions
- Purpose:
  Make experimental scope precise.
- Must include:
  train / test_ff / test_ood usage and real/fake construction logic.
- Should avoid:
  vague “we used standard datasets” wording if exact split logic is custom.
- Evidence sources:
  training and replay code.
- Status:
  `draft done`

### 6.3 Training Configuration
- Purpose:
  State the actual settings used.
- Must include:
  main hyperparameters of upstream/downstream; seed; validation split mode; threshold search range.
- Should avoid:
  including every abandoned hyperparameter in the main table.
- Evidence sources:
  argparse defaults; saved summaries; head meta.
- Status:
  `draft done`

### 6.4 Validation and Threshold Selection
- Purpose:
  Explain how final operating point is chosen.
- Must include:
  holdout method vs within-method split if relevant; threshold search criterion.
- Should avoid:
  mixing post-hoc high-threshold analysis into the official selection protocol without distinction.
- Evidence sources:
  `train_downstream_head.py`, threshold reports.
- Status:
  `draft done`

### 6.5 Evaluation Metrics
- Purpose:
  Define exactly what is reported.
- Must include:
  accuracy, balanced accuracy, fake accuracy, real accuracy, AUC, route top-k if used.
- Should avoid:
  inconsistent metric names across chapters.
- Evidence sources:
  metric functions in training code.
- Status:
  `draft done`

### 6.6 Mainline Variants Compared
- Purpose:
  State which final variants count as core comparisons.
- Must include:
  full mainline; no-FR mainline; possibly threshold-calibrated variants if officially reported.
- Should avoid:
  putting every exploratory branch in the main results table.
- Evidence sources:
  current checkpoints; final reports.
- Status:
  `draft done`

### 6.7 Notes on Fairness / Constraints / Practical Choices
- Purpose:
  Explain choices like cache reuse, sampling limits, holdout strategy.
- Must include:
  any constraints that shaped experiment protocol.
- Should avoid:
  hiding caveats.
- Evidence sources:
  code inventory and open issues tracker.
- Status:
  `draft done`

- Overall Chapter 6 status:
  `draft done`
- Chapter 6 evidence package:
  [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md); [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md); [notes/figures_tables_plan.md](/Users/wuyuchen/Desktop/FYP_final/notes/figures_tables_plan.md); [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); [src/train/train_upstream.py](/Users/wuyuchen/Desktop/FYP_final/src/train/train_upstream.py); [src/train/train_downstream_head.py](/Users/wuyuchen/Desktop/FYP_final/src/train/train_downstream_head.py); [main.py](/Users/wuyuchen/Desktop/FYP_final/main.py); [outputs/replay](/Users/wuyuchen/Desktop/FYP_final/outputs/replay); [chapters/ch6_experimental_setup.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch6_experimental_setup.md).
- Chapter 6 draft now covers:
  experimental questions; dataset and split roles; full vs no-FR setting; two-stage training configuration; validation and threshold selection; evaluation metrics; replay-based mainline evaluation; OOD protocol; and fairness/constraint notes that separate official protocol from later ablation analysis.
- Remaining Chapter 6 needs:
  - finalize which replay/report artifact is treated as the official Chapter 7 main-result source
  - confirm stable dataset/source naming across Chapters 3, 6, and 7
  - decide whether `full` and `no-FR` are presented as co-mainline variants or as final mainline plus major redesign comparison
  - decide whether one compact protocol/setup table should be included in the main text

## Chapter 7 Experimental Results

### 7.1 Mainline Performance on Test-FF
- Purpose:
  Present in-distribution or closer-to-train evaluation.
- Must include:
  main result table and concise interpretation.
- Should avoid:
  drowning this section in ablation detail.
- Evidence sources:
  replay outputs; final summaries.
- Status:
  `not started`

### 7.2 Mainline Performance on OOD / Cross-Domain Evaluation
- Purpose:
  Show the actual challenge setting.
- Must include:
  OOD behaviour; real-side and fake-side trade-offs.
- Should avoid:
  presenting only friendly metrics.
- Evidence sources:
  final replay reports; threshold analyses.
- Status:
  `draft done`

### 7.3 Route-Level Behaviour
- Purpose:
  Explain whether routing provides meaningful structure.
- Must include:
  route probabilities/top1 behaviour if useful; route failures or ambiguity where relevant.
- Should avoid:
  turning into a full ablation section.
- Evidence sources:
  upstream summaries; route-conditioned analysis digests.
- Status:
  `draft done`

### 7.4 Branch-Level Behaviour
- Purpose:
  Explain what patch and pair each contribute.
- Must include:
  branch complementarity narrative.
- Should avoid:
  repeating full feature definitions.
- Evidence sources:
  patch/pair complementarity notes; branch outputs in sample reports.
- Status:
  `draft done`

### 7.5 Final Chosen Variant
- Purpose:
  State clearly what is considered the final system.
- Must include:
  exact chosen pipeline and rationale.
- Should avoid:
  ambiguity between full and no-FR unless intentionally comparing both.
- Evidence sources:
  final wrap-up notes; current repo state.
- Status:
  `draft done`

### 7.6 Result Summary
- Purpose:
  Close results chapter cleanly.
- Must include:
  a short synthesis paragraph.
- Should avoid:
  opening new arguments that belong in discussion.
- Evidence sources:
  all result sections above.
- Status:
  `draft done`

- Overall Chapter 7 status:
  `draft done`
- Chapter 7 evidence package:
  [notes/figures_tables_plan.md](/Users/wuyuchen/Desktop/FYP_final/notes/figures_tables_plan.md); [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md); [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); [notes/student_contributions.md](/Users/wuyuchen/Desktop/FYP_final/notes/student_contributions.md); [outputs/replay](/Users/wuyuchen/Desktop/FYP_final/outputs/replay); [outputs/reports](/Users/wuyuchen/Desktop/FYP_final/outputs/reports); [chapters/ch7_results.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch7_results.md).
- Chapter 7 draft now covers:
  main performance framing; upstream routing behaviour; downstream fusion interpretation; no-FR main-result positioning; OOD and real-shift interpretation; final chosen variant; and a concise result-summary close tied back to project objectives.
- Remaining Chapter 7 needs:
  - finalize the official quantitative source artifact for the main results table
  - insert the finalized quantitative table(s) and any compact comparison figure after the official artifact choice is fixed
  - decide whether Chapter 7 uses one combined main-results table or split tables for `test_ff` and OOD
  - decide whether `no-FR` is presented as the retained final mainline or as the strongest near-final comparison
  - determine whether Section 7.2 gets a compact route-behaviour visual or remains prose-led

## Chapter 8 Ablation Studies and Analysis

### 8.1 Fusion Family Evolution
- Purpose:
  Show the project explored multiple fusion designs before convergence.
- Must include:
  route-only / patch-only / pair-only / multi-fusion families; what survived.
- Should avoid:
  spending too much space on abandoned variants with no lesson.
- Evidence sources:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); backup `src/eval`; repro JSON; old reports.
- Status:
  `outline done`

### 8.2 Pair-Region Subset Ablation
- Purpose:
  Show semantic-region design analysis.
- Must include:
  all_regions vs canonical vs no_background_keep_hair; nuanced conclusion.
- Should avoid:
  overselling tiny end-to-end differences.
- Evidence sources:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); pair-region reports.
- Status:
  `draft done`

### 8.3 Threshold Sweep and Calibration Trade-Off
- Purpose:
  Highlight robustness trade-offs.
- Must include:
  validation threshold vs high-threshold observations; real/fake trade-off.
- Should avoid:
  implying post-hoc threshold sweeps were the default training rule.
- Evidence sources:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); threshold reports.
- Status:
  `draft done`

### 8.4 no-FR Ablation
- Purpose:
  Present the most important late-stage design change.
- Must include:
  head-only no-FR vs end-to-end no-FR distinction; what improved and what was lost.
- Should avoid:
  presenting no-FR as universally better without caveat.
- Evidence sources:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); no-FR reports and current mainline code.
- Status:
  `draft done`

### 8.5 Route-Conditioned Rule / Strategy Search
- Purpose:
  Show systematic search for simpler decision strategies.
- Must include:
  why heuristic strategies were explored and why they were not retained.
- Should avoid:
  too much implementation detail.
- Evidence sources:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); `tmp_route_conditioned_rule_eval_v2.py`, `tmp_strategy_learning_search.py`, corresponding reports.
- Status:
  `draft done`

### 8.6 Alternative Heads and Failed Replacements
- Purpose:
  Demonstrate research depth and negative results.
- Must include:
  correction-aware meta, bridge-gate, soft-gating, sidecars, no-route paired-delta, relative-anchor route.
- Should avoid:
  confusing these with final components.
- Evidence sources:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); backup `experiments/ablations`, `src/eval`, reports.
- Status:
  `draft done`

### 8.7 What Was Revalidated Repeatedly
- Purpose:
  Explicitly prove systematic validation.
- Must include:
  repeated tests on fusion, thresholds, pair regions, no-FR, strategy search.
- Should avoid:
  treating one-off scripts as repeated evidence.
- Evidence sources:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); repeated findings inventory.
- Status:
  `draft done`

- Overall Chapter 8 status:
  `draft done`
- Chapter 8 evidence package:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md); [notes/figures_tables_plan.md](/Users/wuyuchen/Desktop/FYP_final/notes/figures_tables_plan.md); [notes/student_contributions.md](/Users/wuyuchen/Desktop/FYP_final/notes/student_contributions.md); [FYP_final_backup/outputs/reports](/Users/wuyuchen/Desktop/FYP_final_backup/outputs/reports); [FYP_final_backup/src/eval](/Users/wuyuchen/Desktop/FYP_final_backup/src/eval); [FYP_final_backup/experiments/ablations](/Users/wuyuchen/Desktop/FYP_final_backup/experiments/ablations); [outputs/reports](/Users/wuyuchen/Desktop/FYP_final/outputs/reports); [chapters/ch8_ablation_analysis.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch8_ablation_analysis.md).
- Chapter 8 draft now covers:
  fusion-family evolution; pair-region subset analysis; threshold calibration trade-off; no-FR redesign path; heuristic and rule-based alternatives; replaced meta/gating heads; and a final synthesis of repeatedly revalidated findings, all framed around systematic validation and design convergence.
- Remaining Chapter 8 needs:
  - insert finalized tables and figures for `T3`, `T4`, `T5`, and `F5`
  - finalize which secondary ablations remain in main text versus move to appendix
  - decide whether Chapter 8 needs a standalone fusion-family comparison table in addition to the main ablation summary
  - decide whether heuristic/rule-based alternatives warrant a compact main-text table or remain prose plus appendix support

## Chapter 9 Discussion, Contributions and Conclusion

### 9.1 Interpretation of Main Findings
- Purpose:
  Explain what the results mean.
- Must include:
  why route-aware structured fusion helps; what still fails under shift.
- Should avoid:
  repeating raw tables.
- Evidence sources:
  results chapter + ablation chapter.
- Status:
  `not started`

### 9.2 Student's Own Technical Contributions
- Purpose:
  Make authorship and originality explicit.
- Must include:
  concrete contributions in method, engineering, and analysis.
- Should avoid:
  generic statements like “I trained models”.
- Evidence sources:
  contribution tracker below.
- Status:
  `draft done`

### 9.3 Engineering Contributions
- Purpose:
  Make system-building work visible.
- Must include:
  cache architecture, replay pipeline, bundle serialization, cluster organization.
- Should avoid:
  burying engineering under method-only framing.
- Evidence sources:
  implementation chapter notes.
- Status:
  `draft done`

### 9.4 Limitations
- Purpose:
  Be honest and technically precise.
- Must include:
  replay cache dependency; threshold trade-off; limited portability of some artifacts; some gains only partial.
- Should avoid:
  defensive or vague wording.
- Evidence sources:
  open issues tracker; negative result notes.
- Status:
  `draft done`

### 9.5 Future Work
- Purpose:
  Suggest credible next steps.
- Must include:
  stronger domain generalisation, better calibration, cleaner fully portable reproduction, possibly temporal/video extension if justified.
- Should avoid:
  wild future work unrelated to current code.
- Evidence sources:
  limitations + failed attempts.
- Status:
  `draft done`

### 9.6 Final Conclusion
- Purpose:
  End the report clearly.
- Must include:
  problem, method, evidence, contribution, and closing significance.
- Should avoid:
  introducing new experiments.
- Evidence sources:
  final discussion.
- Status:
  `draft done`

- Overall Chapter 9 status:
  `draft done`
- Chapter 9 evidence package:
  [notes/student_contributions.md](/Users/wuyuchen/Desktop/FYP_final/notes/student_contributions.md); [chapters/ch7_results.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch7_results.md); [chapters/ch8_ablation_analysis.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch8_ablation_analysis.md); [chapters/ch5_system_reproducibility.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch5_system_reproducibility.md); [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md); [chapters/ch9_discussion_conclusion.md](/Users/wuyuchen/Desktop/FYP_final/chapters/ch9_discussion_conclusion.md).
- Chapter 9 draft now covers:
  discussion of whether the project objectives were achieved; interpretation of the main findings; the main merits of the proposed solution; explicit student technical contributions; engineering and reproducibility contributions within the wider contribution discussion; limitations; future work; and the final concluding summary.
- Remaining Chapter 9 needs:
  - align the exact “objectives achieved” wording with the final quantitative presentation in Chapter 7 once the main result tables are fixed
  - align the contribution wording with the final phrasing used in Chapter 1 and the Abstract
  - decide in revision whether engineering contributions should remain integrated inside the broader contribution section or be separated again for stylistic consistency
  - add final cross-references once Chapter 7 and Chapter 8 figure/table numbering is fixed

## References
- Purpose:
  Support literature and methods properly.
- Must include:
  deepfake detection, CLIP/backbone, face parsing, geometry/prototype/routing/fusion papers, calibration/generalisation references where used.
- Should avoid:
  incomplete or inconsistent citation formatting.
- Evidence sources:
  to be collected.
- Status:
  `not started`

## Appendices
- Purpose:
  Store material needed for completeness but too detailed for the main narrative.
- Must include:
  extra formulas, extra tables, reproduction commands, additional ablation details, directory/path snapshots if useful.
- Should avoid:
  pushing essential setup requirements entirely out of the main text.
- Evidence sources:
  code inventories, report artifacts, command histories.
- Status:
  `not started`

---

# 4. Mapping from Existing Materials to Report Sections

## Material: Code Inventory / Project-Wide Technical Audit
- Best used in:
  Chapter 4 Proposed Method; Chapter 5 System Implementation and Reproducibility; Chapter 8 Ablation Studies and Analysis.
- Use for:
  identifying mainline modules, distinguishing final vs legacy code, building architecture figures, documenting runtime pipeline.
- Current source basis:
  repository scan of `FYP_final` and `FYP_final_backup`; README; mainline code files.

## Material: Mathematical Formula Extraction
- Best used in:
  Chapter 3 Background and Problem Formulation; Chapter 4 Proposed Method; Appendix A.
- Use for:
  notation table, upstream objective, branch scoring, meta feature vector, threshold rule.
- Current source basis:
  direct extraction from `src/train/train_upstream.py`, `src/train/train_downstream_head.py`, `main.py`, `src/prepare/cache.py`, `src/prepare/face_regions.py`.

## Material: Revalidated Findings Inventory
- Best used in:
  Chapter 8 Ablation Studies and Analysis; Chapter 9 Discussion.
- Use for:
  proving repeated experimentation; supporting discussion of stable findings vs discarded ideas.
- Current source basis:
  backup `outputs/reports`, `tmp_*`, `experiments/ablations`, legacy eval scripts.

## Material: Mainline vs Exploration Classification
- Best used in:
  Chapter 5 System Implementation and Reproducibility; Chapter 8 Ablation Studies and Analysis.
- Use for:
  preventing confusion about what belongs in final system description versus what belongs in ablation/lessons learned.
- Current source basis:
  code audit and repo classification notes.

## Material: outputs/reports/*.md
- Best used in:
  Chapter 7 Results; Chapter 8 Ablation; Chapter 9 Discussion.
- Use for:
  pulling quantitative trends and supporting ablation conclusions.
- Important caution:
  these reports must be filtered into “mainline evidence” vs “exploratory evidence”.

## Material: Replay / Setup / Runtime Information
- Best used in:
  Chapter 5 System Implementation and Reproducibility; Appendix C.
- Use for:
  execution flow, artifact loading, reproduction procedure, threshold/head metadata.
- Current source basis:
  `main.py`, `pipeline_manifest.json`, replay JSON, checkpoint bundles.

## Material: README and Pipeline Manifest
- Best used in:
  Chapter 5 and introduction overview.
- Use for:
  stating which code is active and what the clean repo snapshot represents.

## Material: Cluster Scripts
- Best used in:
  Chapter 5.6 and Appendix C.
- Use for:
  demonstrating experiment orchestration and large-scale preprocessing.

## Material: Backup Ablation Scripts
- Best used in:
  Chapter 8 and selective discussion paragraphs in Chapter 9.
- Use for:
  documenting failed attempts, replacement logic, and research iteration.

---

# 5. Reproducibility / Setup Requirement Tracker

## Why This Section Is Mandatory
The handbook explicitly requires a system setup / reproducibility section. This means the final report must not read like a pure paper-only method report; it must also show how the system was built, organized, and could be rerun.

## Minimum Items That Must Appear in the Main Report
- A clear explanation of repository organization:
  mainline runtime code vs backup research archive.
- A description of the execution pipeline:
  feature extraction, cache generation, upstream training, downstream training, replay evaluation.
- Software environment:
  Python version, key libraries, framework stack.
- Hardware / execution environment:
  local machine, GPU/cluster, or mixed setup as actually used.
- Artifact dependencies:
  what checkpoints and caches are required.
- Reproduction logic:
  what order someone would follow to rerun the mainline.
- Known constraints / caveats:
  especially where clean repo code still depends on existing caches or local artifact layout.

## Items That Should Stay in Main Text
- high-level environment description;
- major runtime components;
- mainline execution order;
- artifact flow;
- one concise reproducibility diagram or table.

## Items That Can Move to Appendix
- long command examples;
- full directory trees;
- exact cluster submission scripts;
- extra configuration snapshots;
- extended path-by-path artifact lists.

## Reproducibility Facts That Must Not Be Forgotten
- `FYP_final` is the cleaned runtime snapshot.
- `FYP_final_backup` contains many exploratory/ablation scripts and historical outputs.
- replay still depends on cache artifacts that may live outside the minimal clean runtime layout.
- downstream replay depends on serialized branch bundles and head metadata.
- compact patch cache is central to practical reruns.

## Current Reproducibility Status
- Main execution path identified:
  yes.
- Mainline scripts identified:
  yes.
- Artifact flow identified:
  yes.
- Exact environment summary written:
  drafted in notes.
- Exact reproduction steps written:
  high-level order drafted in notes.
- Caveat list written:
  yes, in notes.

## Action Items for This Requirement
- Convert [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md) into Chapter 5 prose:
  done at draft level.
- Turn the current environment note into a concise main-text environment table:
  still pending decision.
- Turn the current main execution pipeline note into a Chapter 5 figure or numbered flow:
  still pending figure creation.
- Keep the artifact portability caveat explicit when drafting Chapter 5.7:
  done in current draft.

---

# 6. Student Contribution Tracker

This section exists to prevent the final report from underclaiming or misframing the student’s own work.

## Contribution Category A: Method Design
- Hybrid manifold route model as a structured upstream routing mechanism rather than a plain classifier.
- Integration of route probabilities with downstream evidence rather than isolated branch classification.
- Final downstream design that combines:
  patch branch + pair branch + route-aware meta head.
- no-FR variant as a deliberate architecture-level redesign motivated by robustness observations.

Best chapters:
- Chapter 1 Contributions
- Chapter 4 Proposed Method
- Chapter 9 Discussion / Contributions

## Contribution Category B: Feature and Representation Design
- Facial-region patch aggregation via face parsing.
- Region-level compact cache design that supports downstream training and replay.
- Pair-delta feature formulation using norm + cosine-to-mean-direction descriptors.
- Meta feature construction combining route confidence, uncertainty, branch outputs, and interaction terms.

Best chapters:
- Chapter 4
- Chapter 5
- Appendix A if formula-heavy

## Contribution Category C: Engineering and Experiment Infrastructure
- Three-layer caching design: CLS cache, patch cache, compact patch cache.
- Serialized downstream bundles:
  patch branch, pair branch, route-meta head, head meta.
- Replay runtime that reconstructs evaluation without retraining everything.
- Cluster-oriented experiment organization and helper scripts.

Best chapters:
- Chapter 5
- Chapter 9.3

## Contribution Category D: Experimental and Analytical Work
- Systematic comparison of multiple fusion families.
- Repeated threshold analysis and calibration trade-off studies.
- Pair-region subset ablations.
- no-FR end-to-end ablation.
- Investigation of rule-based, bridge-gated, correction-aware, no-route, and other alternatives.

Best chapters:
- Chapter 8
- Chapter 9

## Contribution Statements That Should Be Repeated Consistently
- “The project contributes a structured mainline system, not only isolated model training.”
- “The student designed and implemented both the modelling pipeline and the experimental infrastructure required to validate it.”
- “The final system emerged from systematic comparison against multiple alternatives, including failed and replaced attempts.”
- “A key contribution of the project is the integration of geometry-aware routing with region-level downstream evidence.”
- “The report distinguishes clearly between final mainline components and exploratory variants.”

## Statements to Avoid
- “I invented a completely new deepfake detector” unless backed carefully.
- “The model solves deepfake detection generally” which overclaims.
- “All explored components improved performance” which is false.
- “The system is fully plug-and-play reproducible” unless the cache dependency issue is resolved and documented.

---

# 7. Writing Order Recommendation

Recommended writing order:

1. **Chapter 5 System Implementation and Reproducibility**
- Reason:
  the codebase is already the strongest evidence base, and this chapter anchors what is actually real in the project.
- Benefit:
  prevents later chapters from describing nonexistent or obsolete pipelines.

2. **Chapter 3 Background and Problem Formulation**
- Reason:
  after implementation is pinned down, the notation and task formulation can be written accurately.

3. **Chapter 4 Proposed Method**
- Reason:
  the method chapter should describe only the confirmed final mainline, and writing it after 3 and 5 reduces drift.

4. **Chapter 6 Experimental Setup**
- Reason:
  once method and implementation are stable, the experiment protocol can be described cleanly.

5. **Chapter 7 Experimental Results**
- Reason:
  results should be tied to clearly defined setup and metrics, not drafted prematurely.

6. **Chapter 8 Ablation Studies and Analysis**
- Reason:
  ablation interpretation is easier once the mainline result narrative is fixed.

7. **Chapter 1 Introduction**
- Reason:
  introduction and contribution claims are easier to write accurately after method/results are clearer.

8. **Chapter 9 Discussion, Contributions and Conclusion**
- Reason:
  this chapter depends on results and ablations being settled.

9. **Abstract**
- Reason:
  always write last so the summary matches the final emphasis.

10. **References and Appendices**
- Reason:
  these grow continuously but can be cleaned near the end.

## Why This Order Is Better Than Writing Intro First
- It reduces speculative wording.
- It helps maintain consistency between claimed method and actual code.
- It prevents omission of the required setup/reproduction section.
- It makes student contribution statements evidence-backed rather than vague.

---

# 8. Open Issues / Missing Materials

## Missing Quantitative Material
- Need a final decision on which exact result tables count as official mainline results.
- Need a final decision on whether the report’s primary final system is:
  full mainline, no-FR mainline, or both compared as final variants.
- Need final numbers/tables to be copied from the chosen replay outputs and reports.

## Missing Figure Material
- Need one overall pipeline diagram.
- Need one upstream route-model diagram or geometry illustration.
- Need one downstream fusion diagram.
- Need one cache / artifact flow diagram for reproducibility chapter.
- Need plots or tables for:
  threshold sweep,
  pair-region ablation,
  no-FR comparison,
  possibly fusion-family evolution summary.

## Missing Environment / Setup Material
- Need actual software environment summary.
- Need actual hardware summary.
- Need exact note on where clean runtime still depends on backup caches or local artifacts.

## Missing Literature Material
- Need curated references for:
  deepfake detection,
  CLIP/frozen vision backbones,
  face parsing,
  prototype/manifold or geometry-based classification,
  routing/fusion/meta-learning or mixture-of-experts style ideas,
  domain shift / calibration / generalisation.

## Missing Evidence Clarifications
- Need explicit confirmation of which scripts produced the final reported numbers.
- Need explicit mapping from specific `outputs/reports/*.md` files to final chapter figures/tables.
- Need a decision on how much legacy `src/eval` content to mention in main text versus appendix.

## Potential Risk Areas During Writing
- Risk of over-describing backup explorations and obscuring the final mainline.
- Risk of under-describing reproducibility and losing handbook marks.
- Risk of presenting no-FR inconsistently across chapters.
- Risk of mixing post-hoc threshold analysis with official validation protocol.
- Risk of weak student-contribution framing if contributions are only implied, not stated.

## Immediate Next Materials to Prepare
- A chapter-level evidence pack for Chapter 5.
- A cleaned formula sheet for Chapters 3-4.
- A “mainline final result shortlist” document.
- A figure checklist with source paths.
- A references shortlist.

## Writing Workspace Skeleton Status
- Chapter skeleton files created under [chapters](/Users/wuyuchen/Desktop/FYP_final/chapters):
  `abstract.md`, `ch1_introduction.md`, `ch2_literature_review.md`, `ch3_background_problem.md`, `ch4_method.md`, `ch5_system_reproducibility.md`, `ch6_experimental_setup.md`, `ch7_results.md`, `ch8_ablation_analysis.md`, `ch9_discussion_conclusion.md`.
- Notes skeleton files created under [notes](/Users/wuyuchen/Desktop/FYP_final/notes):
  `code_inventory.md`, `math_formulas.md`, `repro_setup_notes.md`, `ablation_notes.md`, `figures_tables_plan.md`, `references_todo.md`, `student_contributions.md`.
- Skeleton status interpretation:
  report workspace established; Chapter 4 and Chapter 5 prose drafted; other chapter prose not started.
- Next drafting priority remains:
  Chapter 3 -> Chapter 6 -> Chapter 7, with Chapter 4 and Chapter 5 now available as draft chapters.
- Note completion update:
  [notes/code_inventory.md](/Users/wuyuchen/Desktop/FYP_final/notes/code_inventory.md) is now populated as a detailed technical evidence bank.
- Immediate chapter utility of completed code inventory:
  - Chapter 4:
    final mainline architecture, module roles, route/patch/pair/meta relationships
  - Chapter 5:
    repository structure, artifact flow, cache/replay organization, auxiliary tool boundaries
  - Chapter 8:
    mainline vs exploration distinction, backup ablation source map, repeated-validation evidence locations
- Note completion update:
  [notes/math_formulas.md](/Users/wuyuchen/Desktop/FYP_final/notes/math_formulas.md) is now populated as a detailed formula-material note for the current final mainline.
- Immediate chapter utility of completed math notes:
  - Chapter 4:
    major mathematical material now available for Sections 4.4, 4.5, 4.6, 4.7, and 4.8
  - Chapter 3:
    notation and objective-level material now available for Sections 3.4 and 3.5
  - Appendix A:
    implementation caveats and extra formulas can be lifted selectively if needed
- Note completion update:
  [notes/repro_setup_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/repro_setup_notes.md) is now populated as a setup/reproducibility evidence note.
- Immediate chapter utility of completed reproducibility notes:
  - Chapter 5:
    major material now available for Sections 5.1 through 5.8
  - Appendix C:
    artifact-chain, environment, and command-boundary material now identified
- Note completion update:
  [notes/ablation_notes.md](/Users/wuyuchen/Desktop/FYP_final/notes/ablation_notes.md) is now populated as a consolidated ablation and failed-attempt evidence bank.
- Immediate chapter utility of completed ablation notes:
  - Chapter 8:
    major material now available for Sections 8.1 through 8.7
  - Chapter 9:
    negative-result and trade-off framing now available for discussion/limitations
- Note completion update:
  [notes/figures_tables_plan.md](/Users/wuyuchen/Desktop/FYP_final/notes/figures_tables_plan.md) is now populated as a figure/table planning note for the final report.
- Immediate chapter utility of completed figure/table plan:
  - Chapter 4:
    required architecture figures now scoped for system overview, upstream routing, and downstream fusion
  - Chapter 5:
    reproducibility/runtime pipeline figure now scoped
  - Chapter 7:
    official main-result table and final-variant comparison table now scoped
  - Chapter 8:
    core ablation summary table, threshold plot, pair-region table, and no-FR evolution table now scoped
- Note completion update:
  [notes/references_todo.md](/Users/wuyuchen/Desktop/FYP_final/notes/references_todo.md) is now populated as a reference-planning note for literature review and bibliography preparation.
- Immediate chapter utility of completed reference note:
  - Chapter 2:
    literature categories, benchmark citation needs, and method-positioning reference gaps now scoped
  - Chapter 3:
    dataset and evaluation-reference needs now identified
  - Chapter 4:
    backbone, face parsing, geometry/prototype, and routing-related citation needs now identified
  - Chapter 6 and Chapter 7:
    benchmark/dataset citation checklist now identified
- Note completion update:
  [notes/student_contributions.md](/Users/wuyuchen/Desktop/FYP_final/notes/student_contributions.md) is now populated as a contribution-framing note for later report drafting.
- Immediate chapter utility of completed student contribution note:
  - Chapter 1:
    contribution list and introduction-level contribution wording now scoped
  - Chapter 4:
    method-design contribution framing now clarified without overclaiming novelty
  - Chapter 8:
    analytical contribution and design-iteration contribution framing now clarified
  - Chapter 9:
    student-owned technical, engineering, and experimental contributions now explicitly organized
- Note completion update:
  [notes/report_consistency_audit.md](/Users/wuyuchen/Desktop/FYP_final/notes/report_consistency_audit.md) is now populated as a cross-chapter consistency audit note for final revision.
- Immediate report-wide utility of consistency audit note:
  - whole report:
    objective consistency, contribution consistency, `full` vs `no-FR` positioning, terminology alignment, and mainline/ablation separation issues are now explicitly tracked in one place
  - Chapter 5:
    repo-facing phrasing that should be softened for final prose is now identified
  - Chapters 3/4/6/7:
    notation, split naming, and final-system wording cleanup tasks are now identified
  - Chapters 1/9 and Abstract:
    contribution-wording harmonization tasks are now identified

---

# Maintenance Rules for This File

- Update this file before or immediately after drafting any chapter.
- If a chapter emphasis changes, update both:
  section purpose and contribution tracker.
- If new evidence is discovered, add it to the materials mapping section.
- If a result is demoted from “mainline” to “ablation”, update all affected chapters here before editing prose elsewhere.
- Do not treat this file as a polished report chapter; it is a long-lived control document.
