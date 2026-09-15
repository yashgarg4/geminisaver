# GeminiSaver — developer tasks
# On Windows, run these targets from Git Bash, or copy the command bodies.

VENV_PY := .venv/Scripts/python.exe   # POSIX venvs: .venv/bin/python

.PHONY: help install proxy demo dashboard test

help:
	@echo "make install    - create venv deps (assumes .venv exists & is active)"
	@echo "make proxy      - run the OpenAI-compatible proxy on :8000"
	@echo "make demo       - run the savings demo (Phase 3+)"
	@echo "make dashboard  - run the Streamlit savings dashboard (Phase 4+)"
	@echo "make test       - run the test suite"

install:
	$(VENV_PY) -m pip install -r requirements.txt
	$(VENV_PY) -m pip install -e .

proxy:
	$(VENV_PY) -m uvicorn geminisaver.proxy:app --host 127.0.0.1 --port 8000 --reload

demo:
	$(VENV_PY) examples/demo.py

dashboard:
	$(VENV_PY) -m streamlit run dashboard/app.py

test:
	$(VENV_PY) -m pytest -q
