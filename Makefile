.PHONY: tests clean

tests:
	rm -f tests_log.txt
	make test_utils test_ir test_transformer test_frontend \
		test_matcher test_graph_constructor

test_%:
	@if [ -d .venv ]; then \
		 pipenv run pytest tests/$@ -vv -s | tee -a tests_log.txt; \
	 else \
	 	 pytest tests/$@ -vv -s | tee -a tests_log.txt; \
	 fi;

package:
	rm -rf dist/*
	.venv/bin/python setup.py bdist_wheel sdist
	rm -rf build utensor_cgen.egg-info/

upload-test: package
	.venv/bin/twine upload -r pypitest dist/*

install-test: package
	pip uninstall -y utensor-cgen
	pip install dist/*.tar.gz

upload: package
	.venv/bin/twine upload -r pypi dist/*

clean:
	rm -rf tests_log.txt *.pdf .pytest_cache dist/ build/
