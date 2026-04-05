# FYP Final Report Style Rules

Last updated: 2026-04-02  
Role of this file: persistent style, structure, and writing-control rules for all future FYP final report drafting.  
Usage rule: this file must be read together with [report_master_outline.md](/Users/wuyuchen/Desktop/FYP_final/report_master_outline.md) before writing any report section.

---

# 1. Global Writing Goals

## 1.1 Overall Goal
The final report must read like a **research-oriented academic report**, not like:
- a repository walkthrough,
- a code audit memo,
- a software README,
- or a chronological engineering diary.

## 1.2 What the Report Must Achieve
The report must satisfy the handbook-style expectations that a final-year project report should contain:
- clear objectives;
- clear significance and motivation;
- clear main ideas and technical rationale;
- meaningful method description;
- real experimental results and discussion;
- explicit statement of contributions;
- a proper conclusion;
- a dedicated setup / system implementation / reproducibility section.

## 1.3 How This Project Should Be Presented
This project must be presented as:
- a **deepfake detection research project** with a coherent final pipeline;
- a project containing **non-trivial method design**, not merely standard training;
- a project with **systematic validation and iteration**, not just one model run;
- a project that includes both **algorithmic work** and **reproducible system organization**.

## 1.4 What Counts as Non-Trivial in This Report
The report should consistently show that the work is non-trivial because it includes:
- a hybrid manifold route model rather than a plain classifier;
- a structured downstream design with patch branch, pair branch, and route-aware fusion;
- facial-region aggregation and compact cache engineering;
- repeated ablation and replacement of multiple alternatives;
- explicit consideration of domain shift, threshold trade-off, and no-FR redesign.

## 1.5 Narrative Priority
Whenever there is a choice, prefer writing that emphasizes:
1. objective,
2. rationale,
3. method,
4. evidence,
5. implication.

Do not let the writing be dominated by:
- filenames,
- script chronology,
- parameter dumping,
- or “what command was run first”.

---

# 2. Tone and Style Rules

## 2.1 Required Tone
Use a **formal academic report tone** throughout the final report.

The writing should sound like:
- a technical research paper,
- an FYP dissertation,
- or a serious systems-and-methods report.

It should not sound like:
- an internal audit note,
- a coding assistant summary,
- a lab notebook,
- or an issue tracker.

## 2.2 Sentence Style
Preferred sentence pattern:
- claim -> explanation -> implication.

Good style examples:
- “The upstream routing model was designed to capture structured deviations from the real manifold.”
- “This design allows group-specific evidence to be incorporated before the final binary decision.”
- “The ablation indicates that threshold calibration substantially affects real-side robustness.”

Avoid sentences like:
- “This file does X.”
- “We checked the repo and found Y.”
- “Confirmed from code, the script loads Z.”
- “Mainline or not: yes.”

## 2.3 Forbidden Internal Markers in Final Prose
The following internal analysis labels must never appear in final chapter prose:
- “Confirmed from code”
- “Strongly suggested by code structure”
- “Likely but not fully confirmed”
- “Mainline or not”
- “Category A/B/C”
- “repo inventory”
- “code audit”
- “evidence table” as a phrase unless it is actually a formal appendix table

These labels are acceptable only in working notes, not in report chapters.

## 2.4 Avoid Overly Conversational Style
Do not use:
- “basically”
- “kind of”
- “a lot of”
- “pretty”
- “we just”
- “so”
- “it turns out”
- “as mentioned before” repeatedly

Replace with:
- “primarily”
- “specifically”
- “substantially”
- “the results indicate”
- “the design was motivated by”

## 2.5 Avoid Engineering-Log Style
Do not write the report as:
- “first we tried this, then we changed that script, then we reran…”

If evolution must be discussed, rewrite it as:
- hypothesis,
- design alternative,
- observed limitation,
- revised design,
- conclusion.

## 2.6 Preferred Focus of Every Paragraph
Each paragraph should primarily answer one of these:
- What problem is being addressed?
- Why is this design used?
- How does the method work?
- What do the results show?
- What is learned from the comparison?

If a paragraph is mostly listing files, commands, or directories, it is probably not suitable for the main report text.

---

# 3. Content Allocation Rules

## 3.1 What Belongs in the Main Text
The main text should contain:
- the motivation and problem framing;
- the final method and its rationale;
- the main implementation structure at a system level;
- the reproducibility/setup chapter required by the handbook;
- the official experimental setup;
- the main results;
- the most important ablations;
- the key limitations and discussion;
- the student’s concrete contributions.

## 3.2 What Belongs in the Appendix
The appendix is the right place for:
- long command examples;
- detailed path layouts;
- lengthy parameter dumps;
- extended per-method result tables;
- extra ablation outputs that support but do not drive the main narrative;
- supplementary formulas that are correct but too detailed for the main method chapter;
- large directory structure snapshots;
- additional cluster-script details.

## 3.3 What Must Not Flood the Main Text
The following are usually too detailed for the main chapters unless selectively summarized:
- every script name in backup;
- raw command sequences;
- long path strings;
- exhaustive argparse parameter lists;
- all temporary or exploratory scripts;
- file-by-file implementation inventories.

## 3.4 Setup / Reproduction Placement Rule
The setup / system implementation / reproducibility material must appear in the **main text** as its own clear chapter.

It must not be hidden only in:
- appendix,
- footnotes,
- or scattered short remarks in method/setup sections.

## 3.5 Commands and Paths Rule
Commands, filesystem paths, and long runtime instructions:
- may be mentioned briefly in the main text only when necessary to explain the pipeline;
- should otherwise be moved to appendix, notes, or supplementary material.

## 3.6 Implementation Detail Filter
Before including an implementation detail in the main text, ask:
1. Does it help explain a design choice?
2. Does it affect reproducibility?
3. Does it change the interpretation of results?

If the answer is no to all three, it should probably not be in the main text.

## 3.7 Backup / Legacy Material Rule
Backup or exploratory material may be discussed in the main text only if it serves one of these purposes:
- it explains method evolution;
- it supports an ablation conclusion;
- it reveals an important failure mode;
- it justifies why the final mainline was chosen.

Otherwise, keep it in appendix or omit it.

---

# 4. Contribution Writing Rules

## 4.1 Contribution Must Be Explicit
The report must explicitly identify the student’s own contributions.

Do not rely on the reader to infer contribution from technical detail alone.

## 4.2 Avoid Weak Contribution Language
Avoid generic phrases such as:
- “we implemented the model”
- “we conducted experiments”
- “we tried several methods”

These are too weak unless followed by specifics.

## 4.3 Required Contribution Structure
Whenever contribution is discussed, it should be framed in one or more of these forms:
- **design contribution**:
  what architecture, formulation, or integration was proposed or chosen;
- **implementation contribution**:
  what system, cache, replay, or execution framework was built;
- **validation contribution**:
  what comparisons, ablations, and re-evaluations were carried out;
- **research contribution**:
  what insight, trade-off, or negative result was established.

## 4.4 Contribution Statements Must Be Concrete
Good contribution statements should mention actual technical content, for example:
- design of hybrid manifold routing for coarse fake-group estimation;
- integration of route-aware downstream fusion using patch and pair evidence;
- development of compact region-level caches to support repeated experimentation;
- systematic analysis of no-FR, threshold calibration, and pair-region variants.

Avoid vague statements like:
- “The project contributes to deepfake detection.”
- “The project is meaningful.”
- “A robust system was developed.”

## 4.5 Contribution Placement Rule
Contributions must appear in at least these locations:
- Introduction: concise contribution list
- Method chapter: implicit through design explanation
- Discussion / Contributions chapter: explicit restatement

## 4.6 Ownership Framing Rule
When appropriate, the report should make clear that the student:
- led the design of the final mainline;
- implemented the key training and replay pipeline;
- performed and interpreted the ablation studies;
- converged from multiple alternatives to a final structured system.

This should be stated professionally, not boastfully.

---

# 5. Method Writing Rules

## 5.1 Method Chapter Purpose
The Method chapter must explain:
- the modelling assumptions,
- the architecture,
- the relationship between modules,
- and why the final design takes its current form.

It should not read like a source-code dump.

## 5.2 Formula Selection Rule
Include only the formulas that are central to understanding the method:
- upstream routing logits and objectives;
- branch feature definitions when they carry methodological meaning;
- route-aware fusion equations;
- final threshold rule.

Do not include every code-level transformation if it does not help the reader understand the method.

## 5.3 Formula Explanation Rule
Every important formula must be followed by:
- variable definitions,
- an explanation of what the formula is doing,
- and its intuition or role in the system.

Do not place formulas without commentary.

## 5.4 Method Chapter Should Emphasize
- the hypothesis behind hybrid manifold routing;
- why facial regions are used;
- why patch and pair provide complementary evidence;
- why route-aware fusion is preferable to flat combination;
- why the no-FR variant was investigated.

## 5.5 Method Chapter Should Avoid
- raw script names as the main narrative;
- detailed checkpoint filenames;
- full parameter tables;
- all abandoned alternatives;
- post-hoc result interpretation.

## 5.6 Mainline Purity Rule
The Method chapter must describe the **final mainline** first and cleanly.

Alternative or failed methods may be mentioned briefly only when:
- they are needed to motivate the final choice, and
- they are clearly labeled as alternatives rather than main components.

## 5.7 Code-to-Method Compression Rule
When turning code into method prose:
- compress implementation detail into conceptual operations;
- keep module names only when useful;
- convert file-level behaviour into algorithm-level explanation.

Example:
- not “the script loads `pair_bundle[\"pair_mean_dirs\"]`”
- but “the pair branch uses group-specific mean delta directions learned from paired fake-real references”

---

# 6. Results and Ablation Writing Rules

## 6.1 Results Must Be Interpreted
Results sections must not be just table dumps.

Every major table or figure should be followed by text answering:
- what changed,
- why it matters,
- and what it implies for the project objective.

## 6.2 Main Results Rule
Main results should focus on:
- the final chosen mainline;
- the relevant evaluation settings;
- the main trade-offs;
- and the most important comparison points.

Do not overload the main results section with all backup experiments.

## 6.3 Ablation Chapter Purpose
The Ablation chapter should demonstrate:
- systematic validation,
- design iteration,
- comparative reasoning,
- and evidence-driven convergence.

It should not read like a pile of miscellaneous scripts.

## 6.4 Failed Attempts Rule
Failed attempts may be included only if they have analytical value.

A failed attempt is worth writing when it shows:
- why a simpler alternative did not work,
- why a more complex alternative was not retained,
- what assumption turned out to be weak,
- or what trade-off became visible.

## 6.5 Exploration Integrity Rule
Do not present exploratory or backup scripts as if they were part of the final deployed system.

If an experiment is exploratory:
- say what idea it tested,
- say what conclusion it led to,
- and state that it was not retained.

## 6.6 Revalidation Rule
Ablation writing should explicitly distinguish:
- one-off exploratory tests,
- repeated validations,
- and final retained conclusions.

## 6.7 Result Language Rule
Use evidence-driven phrases such as:
- “The results indicate…”
- “This suggests that…”
- “A consistent pattern across repeated runs is…”
- “The ablation shows that…”

Avoid weak phrases such as:
- “looks better”
- “seems fine”
- “worked well”
- “did not really help”

Replace with more precise interpretations.

## 6.8 Trade-Off Rule
Whenever reporting improvement, also report:
- what remained difficult,
- what metric got worse,
- or what trade-off was introduced.

This is especially important for:
- threshold calibration,
- no-FR,
- real-vs-fake balance under OOD.

---

# 7. Reproducibility Writing Rules

## 7.1 Mandatory Scope
The reproducibility/setup chapter must clearly explain:
- environment,
- data preparation,
- feature extraction,
- cache generation,
- training order,
- evaluation/replay order,
- and artifact dependencies.

## 7.2 Reader Understanding Rule
The chapter must be clear enough that a reader can understand:
- what needs to exist before training,
- what is trained first,
- what is produced,
- and how evaluation is rerun.

The goal is not necessarily one-command reproducibility, but reproducibility **understood and documented honestly**.

## 7.3 Honesty Rule
If the current project still depends on historical caches, local artifacts, or previously generated files, this must be stated explicitly.

Do not claim full portability if the codebase does not currently support it cleanly.

## 7.4 What Must Be Included
At minimum, write:
- software stack summary;
- hardware/execution environment summary;
- cache/artifact flow;
- script-level execution order at a high level;
- how checkpoints and branch bundles are reused in replay.

## 7.5 What Should Be Deferred to Appendix
- long shell commands;
- repeated command variants;
- exact cluster submission details;
- large environment dumps;
- full path inventories.

## 7.6 Reproducibility Framing Rule
This chapter should be written as:
- system design for reproducible experimentation,
not as:
- “here are some random commands I ran”.

## 7.7 Current Project-Specific Reminder
For this project, the report must explicitly distinguish:
- clean final runtime snapshot;
- backup archive with exploratory and historical material;
- serialized branch bundles;
- and cache dependency as a real practical consideration.

---

# 8. Cross-Chapter Consistency Rules

## 8.1 Introduction Rule
Introduction should cover:
- motivation,
- objective,
- challenge,
- contribution,
- and a brief system overview.

It should not contain:
- full mathematical derivations,
- full experiment tables,
- or long repository description.

## 8.2 Literature Review Rule
Literature Review should:
- review prior work,
- identify gaps,
- and motivate the project’s direction.

It should not:
- spend large space describing the current project’s implementation.

## 8.3 Method Rule
Method should:
- explain the final design and rationale.

Method should not:
- contain long experimental interpretation,
- or a full catalogue of discarded alternatives.

## 8.4 Experimental Setup Rule
Experimental Setup should:
- define data splits,
- metrics,
- validation protocol,
- training settings,
- comparison scope.

It should not:
- include long result interpretation,
- or broad discussion paragraphs.

## 8.5 Results Rule
Results should:
- present findings and immediate interpretation.

They should not:
- re-explain the full method,
- or duplicate setup details already defined.

## 8.6 Ablation Rule
Ablation should:
- explain what was changed and why,
- what happened,
- and what was learned.

It should not:
- repeat all mainline architecture description from scratch.

## 8.7 Discussion / Conclusion Rule
Discussion and conclusion must:
- return to the project objectives,
- state contributions clearly,
- summarize what was achieved,
- and acknowledge limitations.

They should not:
- introduce brand-new unexplained technical content.

## 8.8 Anti-Repetition Rule
If the same point appears in multiple chapters, each occurrence must serve a different purpose.

Example:
- Introduction:
  mention that route-aware fusion is a key idea.
- Method:
  explain how route-aware fusion works.
- Results:
  show what it achieved.
- Discussion:
  interpret why it mattered and where it still failed.

Do not mechanically restate the same paragraph in all chapters.

## 8.9 Terminology Consistency Rule
Use the same terminology throughout the report for:
- hybrid manifold route model
- patch branch
- pair branch
- route-aware meta head
- full vs no-FR variant
- test_ff vs OOD / cross-domain evaluation

Do not rename the same component differently in different chapters unless intentionally justified.

---

# 9. Forbidden Patterns

The following are common failure modes that must be avoided in this project.

## 9.1 Mixing Mainline and Backup Exploration
Do not write as if every backup script belongs to the final pipeline.

Wrong:
- describing bridge-gate, correction-aware, and route-meta as if they were all final modules.

Correct:
- describe final mainline first,
- then discuss the others as alternatives or failed replacements in ablation.

## 9.2 Omitting Setup / Reproducibility
Do not leave setup/reproducibility implicit.

If the final report does not contain a clear main-text section on system setup and reproducibility, marks may be lost.

## 9.3 Moving Too Much Essential Content to Appendix
Do not push essential method logic, setup requirements, or key ablation conclusions into appendix only.

Appendix is for support, not for hiding core content.

## 9.4 Using “Many Experiments Were Conducted” as a Substitute for Analysis
Do not write:
- “many experiments were carried out”
- “a large number of tests were done”

unless the report then specifies:
- what was varied,
- what was observed,
- and what conclusion followed.

## 9.5 Explaining Only Implementation, Not Rationale
Do not write large sections that only say:
- what file does what,
- what classifier was trained,
- what output was saved,
without also explaining:
- why this component exists,
- what hypothesis it encodes,
- how it contributes to the pipeline.

## 9.6 Explaining Only Method, Not Trade-Off
Do not present the method as if it has no limitations.

This project must explicitly discuss:
- domain-shift difficulty,
- threshold trade-off,
- incomplete portability of some artifacts,
- and limits of certain improvements.

## 9.7 Overclaiming Robustness or Novelty
Avoid unsupported claims such as:
- “the proposed model solves domain shift”
- “the framework is universally robust”
- “the approach outperforms all prior work”

State only what the project evidence supports.

## 9.8 Treating Failed Attempts as Embarrassing Noise
Do not omit failed attempts if they are analytically useful.

For this project, some failed or replaced designs are valuable evidence of research depth.

## 9.9 Writing the Report as a Directory Tour
Avoid paragraphs dominated by:
- folder names,
- filenames,
- command names,
- path strings,
- or joblib bundle keys.

These belong only where they support reproducibility or evidence.

## 9.10 Copying Internal Analysis Language into Final Prose
Do not paste note-style language from internal audits into the report.

Examples to avoid:
- “mainline or not”
- “category B exploration”
- “confirmed directly from code”
- “likely but not fully confirmed”

---

# 10. Editing Workflow Rule

These workflow rules are mandatory for all future report-writing turns.

## 10.1 Pre-Writing Read Order
Before writing any chapter or subsection, always:
1. read [report_master_outline.md](/Users/wuyuchen/Desktop/FYP_final/report_master_outline.md)
2. read [report_style_rules.md](/Users/wuyuchen/Desktop/FYP_final/report_style_rules.md)

Do not start drafting without refreshing both documents.

## 10.2 Scope Rule
Each writing turn should handle only:
- the currently specified chapter,
- section,
- or tightly scoped subsection.

Do not silently expand into writing multiple unrelated chapters.

## 10.3 Mainline Control Rule
Before drafting, confirm:
- whether the target section is about final mainline,
- ablation,
- setup/reproducibility,
- discussion,
- or appendix.

Then keep the content within that category.

## 10.4 Post-Writing Update Rule
After drafting a section, update [report_master_outline.md](/Users/wuyuchen/Desktop/FYP_final/report_master_outline.md) to reflect:
- the new status of the section,
- any newly identified evidence sources,
- any unresolved questions,
- any changes in chapter emphasis.

## 10.5 Evidence Integrity Rule
If a draft introduces a claim not already grounded in:
- code,
- formula notes,
- experiment reports,
- or accepted literature,
that claim must be flagged for verification before it is allowed to remain in the report draft.

## 10.6 Revision Rule
When revising a drafted section:
- preserve the chapter’s purpose from `report_master_outline.md`;
- enforce the tone rules in this file;
- remove internal-analysis language;
- reduce repetition with earlier chapters.

## 10.7 Consistency Rule
If terminology or contribution framing changes in one drafted section, update:
- the relevant chapter plan in `report_master_outline.md`,
- and any future dependent sections as needed.

## 10.8 Project-Specific Practical Rule
For this FYP, every new draft should explicitly check whether it is:
- accidentally treating backup exploration as final mainline,
- missing setup/reproduction information,
- or failing to state the student’s own contribution concretely.

---

# Maintenance Rule for This File

- Update this file whenever writing quality drifts or recurring problems appear.
- If a new stylistic failure mode is discovered during drafting, add it to Section 9.
- If chapter boundaries become unclear, reinforce them in Section 8.
- This file is a long-term constraint document, not part of the final submitted report.

