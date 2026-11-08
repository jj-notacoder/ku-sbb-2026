# AI disclosure

This is a two-person entry submitted as team **JJSS**. We used Claude through Claude Code for most of the implementation and experiment execution. Gemini CLI was also used during the early exploratory phase.

## Human direction

The human briefs set the project goals, experiment plans, acceptance criteria, time budgets and stop rules. The session log also records human instructions to continue searches, resume interrupted runs and manage memory constraints.

One team member directed the modelling and experiment work using the AI tools described below. The other helped with competition submissions and worked jointly on preparing and editing the final report.

## AI assistance

Claude Code wrote most of the feature-extraction, validation, training and reporting code. It proposed statistical approaches, ran long experiments, analysed results and drafted documentation. Within the human briefs, it also made implementation choices and decisions about which candidates to continue or stop.

The session log records these decisions and departures from the briefs. For example, time reversal was initially described as a diagnostic expected to hurt performance. The assistant advanced it after it passed the validation criteria. Some searches were shortened or stopped because of compute limits or weak preliminary results.

## Evidence and checks

Experiment results are recorded in `results/experiment_log.csv`, with the session history in `results/analysis/session_log.md`. The sequence-model, simulated-data and alternative-learner tracks were tested but did not produce sufficient gains for inclusion in the final model. No ROCKET or MiniRocket result is claimed because the available records do not contain one.

The repository records confirmation-fold evaluations, checks on degenerate inputs and prediction invariance, and a full rebuild that reproduced the submitted predictions. The holdout was opened four times during earlier development, so it should not be treated as an untouched final evaluation.
