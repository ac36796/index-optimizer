PYTHON?=python3.10

.PHONY: venv
venv:
	$(PYTHON) -m venv .venv
	./.venv/bin/pip3 install --upgrade pip setuptools wheel pre-commit
	pre-commit sample-config >> .pre-commit-config.yaml
	pre-commit install

.PHONY: update
update: venv
	./.venv/bin/pip3 install -U -r requirements.txt -r requirements-dev.txt

.PHONY: clean
clean:
	rm -rf .venv
