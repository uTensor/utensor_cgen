.PHONY: tests clean

tests:
	rm -f tests_log.txt
	make test_utils test_ir test_transformer test_frontend \
		test_graph_constructor test_backend test_matcher test_api

test_%:
	@if [ -d .venv ]; then \
		 pipenv run pytest tests/$@ -m 'not slow_test and not deprecated and not beta_release' -vv -s | tee -a tests_log.txt; \
	 else \
	 	 pytest tests/$@ -vv -s -m 'not slow_test and not deprecated and not beta_release' | tee -a tests_log.txt; \
	 fi;

package:
	rm -rf dist/*
	pipenv run python3 setup.py bdist_wheel sdist
	rm -rf build

upload-test: package
	.venv/bin/twine upload -r pypitest dist/*

install-test: package
	pip uninstall -y utensor-cgen
	pip install dist/*.tar.gz

upload: package
	.venv/bin/twine upload -r pypi dist/*

clean:
	rm -rf tests_log.txt *.pdf \
	models constants data \
	tests/test_backend/{models,data} \
	.pytest_cache dist/ build/
