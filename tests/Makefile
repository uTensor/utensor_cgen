TESTS=add add_2 add_3 \
	  argmax const matmul linreg placeholder relu \
	  min_1 min_1_1 min_2 min_3 min_3_1 \
	  max_1_1 max_1 max_2 max_3 max_3_1 \
	  max_pool_1 max_pool_2 max_pool_3 \
	  max_pool_4 \
	  reshape_2 reshape_3 reshape_4 \
	  reshape_5 reshape_6 reshape_7 \
	  conv
	  # uTensor does not support reshape_1 yet
.PHONY: $(TESTS)

all:
	make $(TESTS)

$(TESTS):
	@cd $@; \
	 if [ ! -e cpp ]; then \
	 	mkdir cpp; \
	 fi; \
	 cd cpp; \
	 if [ ! -e idx_data ]; then \
	 	mkdir idx_data; \
	 fi; \
	 utensor-cli ../test_quant_$@.pb -m models -d idx_data -o $@_ctx.cpp; \
	 for file in ../output*.idx ; \
	 do \
	 	if [ -e $$file ]; then \
	 		cp $$file idx_data/; \
		fi; \
	 done; \
	 for file in ../input*.idx ; \
	 do \
		if [ -e $$file ]; then \
	 		cp $$file idx_data/; \
		fi; \
	 done
	@if [ "$$COPY" == "true" ]; then \
		make -e TARGET=$@ cp; \
	 fi

cp:
	cp $(TARGET)/cpp/models/*.{cpp,hpp} $(TARGET)/main.cpp ../../uTensor

clean:
	@for dname in $(TESTS); \
	 do \
	 	 cd $$dname; \
		 echo "removing $$dname/cpp"; \
		 rm -rf cpp; \
		 cd - > /dev/null 2>&1; \
	 done
	rm -rf linreg/chkps
