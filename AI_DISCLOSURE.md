# AI disclosure

**Team and planning.** This is a two-person entry, submitted as team **JJSS**.

The first author planned and directed the project:
- set the objective of each work session;
- defined the acceptance criteria a change had to meet before it could ship;
- chose what to attempt and when to stop;
- ran the main experiment track: time-reversal symmetry, reference-model evidence and score tests;
- made every decision about what went into the final submission.

The second team member developed and tested three alternative tracks:
- **sequence models**: a temporal CNN on the raw series, 0.744 alone, and 0.834 with tabular features but with a
  rank correlation of 0.92 to the main model;
- **training on simulated series**: 0.780 on real data, and −0.0072 when used for pre-training;
- **alternative learners and stacking**: XGBoost, extra-trees, LightGBM variants and a seven-model stack, all with
  rank correlations of 0.94–0.99 to the main model, and a best blend of +0.0002.

These were evaluated under the same acceptance rule and did not reach its threshold. Their results are logged in
`results/experiment_log.csv`.

**AI use.** Claude, used through Claude Code, did the heavy implementation work and ran experiments autonomously
within the constraints we set:
- it wrote the bulk of the feature-extraction, cross-validation, training and reporting code to specification;
- it executed long unattended runs;
- it proposed candidate statistical approaches, most of which were rejected on measured evidence when they failed the
  acceptance gate;
- it drafted this documentation.

An earlier exploratory phase used the Gemini CLI in the same way.
