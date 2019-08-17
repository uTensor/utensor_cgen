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

clean:
	rm -rf tests_log.txt *.pdf .pytest_cache
