TESTS=add argmax const matmul linreg placeholder relu \
	  min_1 min_1_1 min_2 min_3 min_3_1 \
	  max_1_1 max_1 max_2 max_3 max_3_1
.PHONY: $(TESTS)

all:
	make $(TESTS)

$(TESTS):
	@cd $@; \
	 if [ ! -e cpp ]; then \
	 	mkdir cpp; \
	 fi; \
	 cd cpp; \
	 utensor-cli ../test_quant_$@.pb -d idx_data -o $@_ctx.cpp; \
	 if [ -e output*.idx ]; then \
	 	cp -f ../output_*.idx idx_data; \
	 fi; \
	 if [ "$@" == "linreg" ]; then \
	 	cp ../*.idx idx_data; \
	 fi

clean:
	@for dname in $(TESTS); \
	 do \
	 	 cd $$dname; \
		 echo "removing $$dname/cpp"; \
		 rm -rf cpp; \
		 cd - > /dev/null 2>&1; \
	 done
	rm -rf linreg/chkps
