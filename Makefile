PHP ?= php
PHPIZE ?= phpize
PHP_CONFIG ?= php-config
PHP_PREFIX ?=
ORT_VERSION ?= 1.31.1
PIE := $(CURDIR)/pie

UNAME_S := $(shell uname -s)
WINDOWS := $(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S))$(findstring CYGWIN,$(UNAME_S))
ifeq ($(UNAME_S),Darwin)
ORT ?= $(CURDIR)/deps/onnxruntime-osx-arm64-$(ORT_VERSION)
endif
ifneq ($(WINDOWS),)
EXT := $(PIE)/modules/php_onnxruntime.dll
else
EXT := $(PIE)/modules/onnxruntime.so
endif

.PHONY: build test models examples bench clean

build:
	mkdir -p target
	$(PHP) tools/ext-version.php > target/ext-version
	$(PHP) -r '$$v = trim(file_get_contents("target/ext-version")); file_put_contents("pie/version.h", "#define PHP_ONNXRUNTIME_VERSION \"" . $$v . "\"\n");'
ifeq ($(WINDOWS),)
	cd $(PIE) && $(PHPIZE) && \
		args="--with-onnxruntime=$(ORT)"; \
		if command -v '$(PHP_CONFIG)' >/dev/null 2>&1; then args="--with-php-config=$(PHP_CONFIG) $$args"; fi; \
		./configure $$args && $(MAKE) -j4
else
	cl_bin=$$(command -v cl || command -v cl.exe); \
	test -n "$$cl_bin"; \
	PATH="$$(dirname "$$cl_bin"):$$PATH"; \
	cd $(PIE) && cmd //c "phpize && configure --with-onnxruntime=$(ORT) --with-prefix=$(PHP_PREFIX) && nmake"
	mkdir -p $(PIE)/modules
	dll=$$(find $(PIE) -name php_onnxruntime.dll -type f ! -path '*/modules/*' | head -n 1); \
	test -n "$$dll"; \
	cp "$$dll" $(EXT)
endif

# The script overwrites what it writes but cannot remove a model a past version
# wrote, so an old tree keeps a file that no longer loads. Clear both first.
models:
	rm -rf tests/models examples/models
	$(PHP) tests/make_models.php

test: build models
	@test -n "$(ORT)" || { echo "ORT is unset. Point it at an ONNX Runtime prefix." >&2; exit 1; }
	ext="$(EXT)"; \
	if [ ! -f "$$ext" ]; then \
		ext=$$(find "$(PIE)" \( -name php_onnxruntime.dll -o -name onnxruntime.dll \) -print | head -1); \
	fi; \
	test -n "$$ext" && test -f "$$ext"; \
	printf '%s\n' "$$ext" > target/ext-library; \
	$(PHP) -r '$$p = $$argv[1]; foreach (["lib/libonnxruntime.1.dylib", "lib/libonnxruntime.so.1", "lib/onnxruntime.dll"] as $$name) { $$path = $$p . "/" . $$name; if (file_exists($$path)) { echo $$path, "\n"; exit(0); } } fwrite(STDERR, "ONNX Runtime shared library not found under $$p/lib\n"); exit(1);' -- "$(ORT)" > target/ort-library; \
	$(PHP) -d extension="$$ext" tests/api.php && \
	$(PHP) -d extension="$$ext" tests/gaps.php && \
	$(PHP) -d extension="$$ext" tests/surface.php && \
	$(PHP) -d extension="$$ext" tests/signatures.php

examples: models
	$(PHP) -d extension=$(EXT) examples/inspect.php
	$(PHP) -d extension=$(EXT) examples/embed.php
	$(PHP) -d extension=$(EXT) examples/reuse.php
	$(PHP) -d extension=$(EXT) examples/image.php
	$(PHP) -d extension=$(EXT) examples/classify.php

bench: build
	@test -x bench/.venv/bin/python || { echo "no bench/.venv. See bench/README.md." >&2; exit 1; }
	bench/.venv/bin/python bench/gen_models.py
	$(PHP) -d extension=$(EXT) bench/run.php

clean:
	cd $(PIE) && phpize --clean || true
