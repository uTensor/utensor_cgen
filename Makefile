.PHONY: tests clean

tests:
	rm -f tests_log.txt
	make test_utils test_ir test_transformer

test_%:
	pipenv run pytest tests/$@ -vv -s | tee -a tests_log.txt

clean:
	rm -f tests_log.txt