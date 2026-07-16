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

.PHONY: test smoke vis

test:                       ## run the unit tests (ROS path neutralized)
	PYTHONPATH= $(PY) -m pytest tests/ -v

smoke:                      ## quick "does the loader work" check
	$(PY) scripts/smoke_load.py

vis:                        ## write annotated RGB|thermal panels to outputs/vis/
	$(PY) scripts/visualize_pairs.py
