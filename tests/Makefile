TESTS=add argmax const matmul
.PHONY: $(TESTS)

all:
	make $(TESTS)

$(TESTS):
	@cd $@; \
	 if [ ! -e cpp ]; then \
	 	mkdir cpp; \
	 fi; \
	 cd cpp; \
	 utensor-cli ../test_quant_$@.pb -o $@_ctx.cpp

clean:
	@for dname in $(TESTS); \
	 do \
	 	 cd $$dname; \
		 echo "removing $$dname/cpp"; \
		 rm -rf cpp; \
		 cd - > /dev/null 2>&1; \
	 done
