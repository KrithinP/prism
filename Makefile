# PRISM convenience targets.
#
# Why the `PYTHONPATH=` prefix on test targets:
# Krithin's shell sources ROS 2 Humble, which exports its Python 3.10 packages
# onto PYTHONPATH. pytest auto-discovers plugins from anything on that path, trips
# over a stale ROS `launch_testing` plugin, and crashes before running a single
# test. Clearing PYTHONPATH for the pytest process removes the ROS packages from
# the path (our venv doesn't use PYTHONPATH), so this is surgical: real plugins
# still load, ROS pollution does not.

PY = .venv/bin/python

.NOTPARALLEL: study-all study-discovery study-operators study-robustness \
	study-diagnostics

.PHONY: test smoke vis prepare-kaist certify-evaluator preflight data-stats \
	matrix matrix-late select-graph matrix-operators select-operator \
	matrix-robustness study-discovery study-operators study-robustness \
	study-diagnostics diagnostics-profiles diagnostics-mechanisms \
	diagnostics-bootstrap diagnostics-summary study-all audit-ledger summary

test:                       ## run the unit tests (ROS path neutralized)
	PYTHONPATH= $(PY) -m pytest tests/ -v

smoke:                      ## quick "does the loader work" check
	$(PY) scripts/smoke_load.py

vis:                        ## write annotated RGB|thermal panels to outputs/vis/
	$(PY) scripts/visualize_pairs.py

prepare-kaist:              ## download/convert/audit the canonical data protocol
	PYTHONPATH= $(PY) scripts/prepare_kaist.py --download

certify-evaluator:          ## persist fail-closed evaluator parity certificate
	PYTHONPATH= $(PY) scripts/parity_check.py

preflight:                  ## validate data, hashes, protocol, evaluator, and CUDA
	PYTHONPATH= $(PY) scripts/preflight.py

data-stats:                 ## train-cv-only means/stds for controlled interventions
	PYTHONPATH= $(PY) scripts/compute_dataset_stats.py

matrix:                     ## core baseline and learned fusion-family blocks
	PYTHONPATH= $(PY) scripts/run_experiments.py --block A
	PYTHONPATH= $(PY) scripts/run_experiments.py --block B

matrix-late:                ## paired-seed union-NMS and ProbEn validation controls
	PYTHONPATH= $(PY) scripts/run_late_fusion_matrix.py

select-graph:               ## freeze the five-seed Block B validation winner
	PYTHONPATH= $(PY) scripts/select_winner.py \
		--arms B2 B3 B4 B5 B6 B7 B8 B9 B10 \
		--out outputs/selection/block_b_winner.json

matrix-operators:           ## C1-C12 on the frozen Block B graph
	PYTHONPATH= $(PY) scripts/run_experiments.py --block C \
		--winning-graph-manifest outputs/selection/block_b_winner.json

select-operator:            ## freeze the five-seed Block C validation winner
	PYTHONPATH= $(PY) scripts/select_winner.py \
		--arms C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 \
		--out outputs/selection/block_c_winner.json

matrix-robustness:          ## D1-D3 on the frozen Block C architecture
	PYTHONPATH= $(PY) scripts/run_experiments.py --block D \
		--winning-graph-manifest outputs/selection/block_c_winner.json

study-discovery: preflight matrix matrix-late select-graph
	PYTHONPATH= $(PY) scripts/run_experiments.py --arm B13 \
		--winning-graph-manifest outputs/selection/block_b_winner.json

study-operators: matrix-operators select-operator

study-robustness: matrix-robustness

diagnostics-profiles:             ## params/MAC/latency for every learned graph
	PYTHONPATH= $(PY) scripts/run_posthoc_studies.py --suite profiles

diagnostics-mechanisms: data-stats ## interventions, representations, attribution
	PYTHONPATH= $(PY) scripts/run_posthoc_studies.py --suite mechanisms

diagnostics-bootstrap:            ## paired validation-sequence uncertainty
	PYTHONPATH= $(PY) scripts/run_posthoc_studies.py --suite bootstrap

diagnostics-summary: diagnostics-profiles diagnostics-mechanisms diagnostics-bootstrap
	PYTHONPATH= $(PY) scripts/summarize_posthoc.py

study-diagnostics: diagnostics-summary

study-all: study-discovery data-stats study-operators study-robustness summary \
	study-diagnostics

audit-ledger:               ## verify lifecycle and every referenced artifact hash
	PYTHONPATH= $(PY) scripts/audit_ledger.py

summary: audit-ledger       ## seed-level tables and paired 95% t intervals
	PYTHONPATH= $(PY) scripts/summarize_results.py
