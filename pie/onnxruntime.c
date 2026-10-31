#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "zend_exceptions.h"
#include "zend_enum.h"
#include "ext/standard/info.h"
#include "php_onnxruntime.h"

#include <ctype.h>
#include <string.h>

const OrtApi *ort_api = NULL;
OrtEnv *ort_env = NULL;
OrtMemoryInfo *ort_cpu_mem = NULL;
OrtAllocator *ort_allocator = NULL;
zend_class_entry *onnx_exception_ce = NULL;
zend_class_entry *onnx_value_ce = NULL;

static zend_class_entry *onnx_runtime_ce = NULL;
zend_class_entry *onnx_options_ce = NULL;
static zend_class_entry *onnx_optimization_ce = NULL;
static zend_class_entry *onnx_log_level_ce = NULL;
static zend_object_handlers onnx_options_handlers;

int onnx_ort_ok(OrtStatus *status)
{
	const char *message;
	char *copy;

	if (status == NULL) {
		return 1;
	}
	message = ort_api->GetErrorMessage(status);
	copy = estrdup(message != NULL ? message : "ONNX Runtime error");
	ort_api->ReleaseStatus(status);
	zend_throw_exception(onnx_exception_ce, copy, 0);
	efree(copy);
	return 0;
}

void onnx_throw(const char *message)
{
	zend_throw_exception(onnx_exception_ce, message, 0);
}

void onnx_throw_feed(const char *message, const char *input, zval *expected, zval *actual)
{
	zend_object *object = zend_throw_exception(onnx_exception_ce, message, 0);

	if (object == NULL) {
		return;
	}
	if (input != NULL) {
		zend_update_property_string(onnx_exception_ce, object, "inputName", sizeof("inputName") - 1, input);
	}
	if (expected != NULL) {
		zend_update_property(onnx_exception_ce, object, "expectedShape", sizeof("expectedShape") - 1, expected);
	}
	if (actual != NULL) {
		zend_update_property(onnx_exception_ce, object, "actualShape", sizeof("actualShape") - 1, actual);
	}
}

const char *onnx_dtype_name(ONNXTensorElementDataType type)
{
	switch (type) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			return "float32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			return "uint8";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			return "int8";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			return "uint16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			return "int16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			return "int32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			return "int64";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
			return "string";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			return "bool";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
			return "float16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			return "float64";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			return "uint32";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
			return "uint64";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
			return "bfloat16";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
			return "float8e4m3fn";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
			return "float8e4m3fnuz";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
			return "float8e5m2";
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
			return "float8e5m2fnuz";
		default:
			return "unknown";
	}
}

int onnx_dtype_by_name(const char *name, size_t len, ONNXTensorElementDataType *type)
{
	if (len == 7 && memcmp(name, "float32", 7) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	} else if (len == 7 && memcmp(name, "float16", 7) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
	} else if (len == 7 && memcmp(name, "float64", 7) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
	} else if (len == 5 && memcmp(name, "int64", 5) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
	} else if (len == 5 && memcmp(name, "int32", 5) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
	} else if (len == 5 && memcmp(name, "int16", 5) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
	} else if (len == 4 && memcmp(name, "int8", 4) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
	} else if (len == 5 && memcmp(name, "uint8", 5) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
	} else if (len == 6 && memcmp(name, "uint16", 6) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
	} else if (len == 6 && memcmp(name, "uint32", 6) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
	} else if (len == 4 && memcmp(name, "bool", 4) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
	} else if (len == 6 && memcmp(name, "string", 6) == 0) {
		*type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
	} else {
		return 0;
	}
	return 1;
}

size_t onnx_dtype_size(ONNXTensorElementDataType type)
{
	switch (type) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			return 4;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
			return 8;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			return 2;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			return 1;
		default:
			return 0;
	}
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_exception_input, 0, 0, IS_STRING, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_exception_shape, 0, 0, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

static PHP_METHOD(OnnxException, inputName)
{
	zval *value;

	ZEND_PARSE_PARAMETERS_NONE();
	value = zend_read_property(onnx_exception_ce, Z_OBJ_P(ZEND_THIS), "inputName", sizeof("inputName") - 1, 1, NULL);
	if (value == NULL || Z_TYPE_P(value) == IS_NULL) {
		RETURN_NULL();
	}
	RETURN_STR_COPY(Z_STR_P(value));
}

static PHP_METHOD(OnnxException, expectedShape)
{
	zval *value;

	ZEND_PARSE_PARAMETERS_NONE();
	value = zend_read_property(onnx_exception_ce, Z_OBJ_P(ZEND_THIS), "expectedShape", sizeof("expectedShape") - 1, 1, NULL);
	if (value == NULL || Z_TYPE_P(value) == IS_NULL) {
		RETURN_NULL();
	}
	RETURN_COPY(value);
}

static PHP_METHOD(OnnxException, actualShape)
{
	zval *value;

	ZEND_PARSE_PARAMETERS_NONE();
	value = zend_read_property(onnx_exception_ce, Z_OBJ_P(ZEND_THIS), "actualShape", sizeof("actualShape") - 1, 1, NULL);
	if (value == NULL || Z_TYPE_P(value) == IS_NULL) {
		RETURN_NULL();
	}
	RETURN_COPY(value);
}

static const zend_function_entry onnx_exception_methods[] = {
	PHP_ME(OnnxException, inputName, arginfo_exception_input, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxException, expectedShape, arginfo_exception_shape, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxException, actualShape, arginfo_exception_shape, ZEND_ACC_PUBLIC)
	PHP_FE_END
};

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_runtime_version, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_runtime_providers, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_runtime_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

zend_string *onnx_provider_short_name(const char *id)
{
	size_t length = strlen(id);
	const char *suffix = "ExecutionProvider";
	size_t suffix_len = strlen(suffix);
	size_t base = length;
	zend_string *short_name;
	size_t i;

	if (length > suffix_len && memcmp(id + length - suffix_len, suffix, suffix_len) == 0) {
		base = length - suffix_len;
	}
	short_name = zend_string_alloc(base, 0);
	for (i = 0; i < base; i++) {
		ZSTR_VAL(short_name)[i] = (char) tolower((unsigned char) id[i]);
	}
	ZSTR_VAL(short_name)[base] = '\0';
	return short_name;
}

static PHP_METHOD(Build, __construct)
{
	ZEND_PARSE_PARAMETERS_NONE();
}

static PHP_METHOD(Build, version)
{
	const OrtApiBase *base;

	ZEND_PARSE_PARAMETERS_NONE();
	base = OrtGetApiBase();
	RETURN_STRING(base->GetVersionString());
}

static PHP_METHOD(Build, availableProviders)
{
	char **providers = NULL;
	int count = 0;
	int i;

	ZEND_PARSE_PARAMETERS_NONE();
	if (!onnx_ort_ok(ort_api->GetAvailableProviders(&providers, &count))) {
		return;
	}
	array_init_size(return_value, count);
	for (i = 0; i < count; i++) {
		zend_string *short_name = onnx_provider_short_name(providers[i]);
		add_next_index_str(return_value, short_name);
	}
	if (!onnx_ort_ok(ort_api->ReleaseAvailableProviders(providers, count))) {
		return;
	}
}

static const zend_function_entry onnx_runtime_methods[] = {
	PHP_ME(Build, __construct, arginfo_runtime_construct, ZEND_ACC_PROTECTED)
	PHP_ME(Build, version, arginfo_runtime_version, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
	PHP_ME(Build, availableProviders, arginfo_runtime_providers, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
	PHP_FE_END
};

static zend_object *onnx_options_new(zend_class_entry *ce)
{
	onnx_options *options = zend_object_alloc(sizeof(onnx_options), ce);

	zend_object_std_init(&options->std, ce);
	object_properties_init(&options->std, ce);
	options->std.handlers = &onnx_options_handlers;
	ZVAL_UNDEF(&options->optimization);
	ZVAL_UNDEF(&options->log_level);
	ZVAL_UNDEF(&options->free_dimensions);
	ZVAL_UNDEF(&options->providers);
	return &options->std;
}

static void onnx_options_free(zend_object *object)
{
	onnx_options *options = onnx_options_from_obj(object);

	if (Z_TYPE(options->optimization) != IS_UNDEF) {
		zval_ptr_dtor(&options->optimization);
	}
	if (Z_TYPE(options->log_level) != IS_UNDEF) {
		zval_ptr_dtor(&options->log_level);
	}
	if (Z_TYPE(options->providers) != IS_UNDEF) {
		zval_ptr_dtor(&options->providers);
	}
	if (Z_TYPE(options->free_dimensions) != IS_UNDEF) {
		zval_ptr_dtor(&options->free_dimensions);
	}
	zend_object_std_dtor(object);
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_options_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_options_thread, 0, 1, IS_STATIC, 0)
	ZEND_ARG_TYPE_INFO(0, threads, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_options_optimization, 0, 1, IS_STATIC, 0)
	ZEND_ARG_OBJ_INFO(0, level, Switon\\OnnxRuntime\\OptimizationLevel, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_options_parallel, 0, 0, IS_STATIC, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, parallel, _IS_BOOL, 0, "true")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_options_log, 0, 1, IS_STATIC, 0)
	ZEND_ARG_OBJ_INFO(0, level, Switon\\OnnxRuntime\\LogLevel, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_options_verbosity, 0, 1, IS_STATIC, 0)
	ZEND_ARG_TYPE_INFO(0, level, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_options_providers, 0, 1, IS_STATIC, 0)
	ZEND_ARG_TYPE_INFO(0, providers, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_options_free_dimensions, 0, 1, IS_STATIC, 0)
	ZEND_ARG_TYPE_INFO(0, dimensions, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

static PHP_METHOD(OnnxOptions, freeDimensions)
{
	onnx_options *options = onnx_options_from_obj(Z_OBJ_P(ZEND_THIS));
	zval *dimensions;
	zend_string *name;
	zval *size;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_ARRAY(dimensions)
	ZEND_PARSE_PARAMETERS_END();
	if (zend_hash_num_elements(Z_ARRVAL_P(dimensions)) == 0) {
		onnx_throw("free dimensions must not be empty");
		return;
	}
	ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL_P(dimensions), name, size) {
		zval *entry = size;

		if (name == NULL || ZSTR_LEN(name) == 0) {
			onnx_throw("a free dimension needs a name");
			return;
		}
		ZVAL_DEREF(entry);
		if (Z_TYPE_P(entry) != IS_LONG) {
			onnx_throw("a free dimension size must be an integer");
			return;
		}
		if (Z_LVAL_P(entry) <= 0) {
			onnx_throw("a free dimension size must be positive");
			return;
		}
	} ZEND_HASH_FOREACH_END();
	if (Z_TYPE(options->free_dimensions) != IS_UNDEF) {
		zval_ptr_dtor(&options->free_dimensions);
	}
	ZVAL_COPY(&options->free_dimensions, dimensions);
	RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

static PHP_METHOD(OnnxOptions, __construct)
{
	ZEND_PARSE_PARAMETERS_NONE();
}

static PHP_METHOD(OnnxOptions, intraOpThreads)
{
	onnx_options *options = onnx_options_from_obj(Z_OBJ_P(ZEND_THIS));
	zend_long count;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_LONG(count)
	ZEND_PARSE_PARAMETERS_END();
	options->thread_count = count;
	options->thread_count_set = 1;
	RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

static PHP_METHOD(OnnxOptions, interOpThreads)
{
	onnx_options *options = onnx_options_from_obj(Z_OBJ_P(ZEND_THIS));
	zend_long count;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_LONG(count)
	ZEND_PARSE_PARAMETERS_END();
	options->parallel_thread_count = count;
	options->parallel_thread_count_set = 1;
	RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

static PHP_METHOD(OnnxOptions, optimization)
{
	onnx_options *options = onnx_options_from_obj(Z_OBJ_P(ZEND_THIS));
	zval *level;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_OBJECT_OF_CLASS(level, onnx_optimization_ce)
	ZEND_PARSE_PARAMETERS_END();
	if (Z_TYPE(options->optimization) != IS_UNDEF) {
		zval_ptr_dtor(&options->optimization);
	}
	ZVAL_COPY(&options->optimization, level);
	RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

static PHP_METHOD(OnnxOptions, parallel)
{
	onnx_options *options = onnx_options_from_obj(Z_OBJ_P(ZEND_THIS));
	zend_bool parallel = 1;

	ZEND_PARSE_PARAMETERS_START(0, 1)
		Z_PARAM_OPTIONAL
		Z_PARAM_BOOL(parallel)
	ZEND_PARSE_PARAMETERS_END();
	options->parallel = parallel;
	options->parallel_set = 1;
	RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

static PHP_METHOD(OnnxOptions, logLevel)
{
	onnx_options *options = onnx_options_from_obj(Z_OBJ_P(ZEND_THIS));
	zval *level;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_OBJECT_OF_CLASS(level, onnx_log_level_ce)
	ZEND_PARSE_PARAMETERS_END();
	if (Z_TYPE(options->log_level) != IS_UNDEF) {
		zval_ptr_dtor(&options->log_level);
	}
	ZVAL_COPY(&options->log_level, level);
	RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

static PHP_METHOD(OnnxOptions, verbosity)
{
	onnx_options *options = onnx_options_from_obj(Z_OBJ_P(ZEND_THIS));
	zend_long level;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_LONG(level)
	ZEND_PARSE_PARAMETERS_END();
	options->verbosity = level;
	options->verbosity_set = 1;
	RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

static PHP_METHOD(OnnxOptions, providers)
{
	onnx_options *options = onnx_options_from_obj(Z_OBJ_P(ZEND_THIS));
	zval *providers;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_ARRAY(providers)
	ZEND_PARSE_PARAMETERS_END();
	if (Z_TYPE(options->providers) != IS_UNDEF) {
		zval_ptr_dtor(&options->providers);
	}
	ZVAL_COPY(&options->providers, providers);
	RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

static const zend_function_entry onnx_options_methods[] = {
	PHP_ME(OnnxOptions, __construct, arginfo_options_construct, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxOptions, intraOpThreads, arginfo_options_thread, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxOptions, interOpThreads, arginfo_options_thread, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxOptions, optimization, arginfo_options_optimization, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxOptions, parallel, arginfo_options_parallel, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxOptions, logLevel, arginfo_options_log, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxOptions, verbosity, arginfo_options_verbosity, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxOptions, providers, arginfo_options_providers, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxOptions, freeDimensions, arginfo_options_free_dimensions, ZEND_ACC_PUBLIC)
	PHP_FE_END
};

static void register_string_enum(zend_class_entry **ce, const char *name, const char *const *cases, const char *const *values, int count)
{
	zend_class_entry *entry;
	int i;

	entry = zend_register_internal_enum(name, IS_STRING, NULL);
	for (i = 0; i < count; i++) {
		zval value;
		zend_string *backing = zend_string_init_interned(values[i], strlen(values[i]), 1);

		ZVAL_STR(&value, backing);
		zend_enum_add_case_cstr(entry, cases[i], &value);
	}
	entry->ce_flags |= ZEND_ACC_FINAL;
	*ce = entry;
}

static void register_exception(void)
{
	zend_class_entry ce;

	INIT_CLASS_ENTRY(ce, "Switon\\OnnxRuntime\\OnnxException", onnx_exception_methods);
	onnx_exception_ce = zend_register_internal_class_ex(&ce, zend_ce_exception);
	zend_declare_property_null(onnx_exception_ce, "inputName", sizeof("inputName") - 1, ZEND_ACC_PRIVATE);
	zend_declare_property_null(onnx_exception_ce, "expectedShape", sizeof("expectedShape") - 1, ZEND_ACC_PRIVATE);
	zend_declare_property_null(onnx_exception_ce, "actualShape", sizeof("actualShape") - 1, ZEND_ACC_PRIVATE);
}

static void register_runtime(void)
{
	zend_class_entry ce;

	INIT_CLASS_ENTRY(ce, "Switon\\OnnxRuntime\\Build", onnx_runtime_methods);
	onnx_runtime_ce = zend_register_internal_class(&ce);
}

static void register_options(void)
{
	zend_class_entry ce;

	INIT_CLASS_ENTRY(ce, "Switon\\OnnxRuntime\\OnnxOptions", onnx_options_methods);
	onnx_options_ce = zend_register_internal_class(&ce);
	onnx_options_ce->create_object = onnx_options_new;
	memcpy(&onnx_options_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
	onnx_options_handlers.offset = XtOffsetOf(onnx_options, std);
	onnx_options_handlers.free_obj = onnx_options_free;
	onnx_options_handlers.clone_obj = NULL;
}

static void declare_value_constant(zend_class_entry *ce, const char *name, const char *value)
{
	zend_declare_class_constant_string(ce, name, strlen(name), value);
}

PHP_RINIT_FUNCTION(onnxruntime)
{
	ZEND_TSRMLS_CACHE_UPDATE();
	return SUCCESS;
}

PHP_MINIT_FUNCTION(onnxruntime)
{
	const OrtApiBase *base;
	const char *optimization_cases[] = {"Off", "Basic", "Extended", "All"};
	const char *optimization_values[] = {"off", "basic", "extended", "all"};
	const char *log_cases[] = {"Verbose", "Info", "Warning", "Error", "Fatal"};
	const char *log_values[] = {"verbose", "info", "warning", "error", "fatal"};

	ZEND_TSRMLS_CACHE_UPDATE();
	base = OrtGetApiBase();
	ort_api = base->GetApi(ORT_API_VERSION);
	if (ort_api == NULL) {
		zend_error(E_WARNING, "onnxruntime: ONNX Runtime does not provide API version %u", ORT_API_VERSION);
		return FAILURE;
	}
	{
		OrtStatus *status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxruntime", &ort_env);
		if (status != NULL) {
			zend_error(E_WARNING, "onnxruntime: %s", ort_api->GetErrorMessage(status));
			ort_api->ReleaseStatus(status);
			return FAILURE;
		}
		status = ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &ort_cpu_mem);
		if (status != NULL) {
			zend_error(E_WARNING, "onnxruntime: %s", ort_api->GetErrorMessage(status));
			ort_api->ReleaseStatus(status);
			return FAILURE;
		}
		status = ort_api->GetAllocatorWithDefaultOptions(&ort_allocator);
		if (status != NULL) {
			zend_error(E_WARNING, "onnxruntime: %s", ort_api->GetErrorMessage(status));
			ort_api->ReleaseStatus(status);
			return FAILURE;
		}
	}

	register_exception();
	register_string_enum(&onnx_optimization_ce, "Switon\\OnnxRuntime\\OptimizationLevel", optimization_cases, optimization_values, 4);
	register_string_enum(&onnx_log_level_ce, "Switon\\OnnxRuntime\\LogLevel", log_cases, log_values, 5);
	register_options();
	onnx_register_values();
	declare_value_constant(onnx_value_ce, "FLOAT32", "float32");
	declare_value_constant(onnx_value_ce, "FLOAT16", "float16");
	declare_value_constant(onnx_value_ce, "INT64", "int64");
	declare_value_constant(onnx_value_ce, "INT32", "int32");
	declare_value_constant(onnx_value_ce, "INT8", "int8");
	declare_value_constant(onnx_value_ce, "UINT8", "uint8");
	declare_value_constant(onnx_value_ce, "BOOL", "bool");
	declare_value_constant(onnx_value_ce, "STRING", "string");
	declare_value_constant(onnx_value_ce, "TENSOR", "tensor");
	declare_value_constant(onnx_value_ce, "SEQUENCE", "sequence");
	declare_value_constant(onnx_value_ce, "MAP", "map");
	onnx_register_sessions();
	register_runtime();
	return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(onnxruntime)
{
	/* The process owns the environment. Releasing it here races the library destructor and corrupts the PHP allocator. */
	ort_cpu_mem = NULL;
	ort_env = NULL;
	ort_api = NULL;
	return SUCCESS;
}

PHP_MINFO_FUNCTION(onnxruntime)
{
	const OrtApiBase *base = OrtGetApiBase();

	php_info_print_table_start();
	php_info_print_table_row(2, "Vendor", "Switon");
	php_info_print_table_row(2, "URL", "https://github.com/switon-php/ext-onnxruntime");
	php_info_print_table_row(2, "Summary", "Runs an ONNX model");
	php_info_print_table_row(2, "Version", PHP_ONNXRUNTIME_VERSION);
	php_info_print_table_row(2, "ONNX Runtime", base->GetVersionString());
	php_info_print_table_end();
}

zend_module_entry onnxruntime_module_entry = {
	STANDARD_MODULE_HEADER,
	"onnxruntime",
	NULL,
	PHP_MINIT(onnxruntime),
	PHP_MSHUTDOWN(onnxruntime),
	PHP_RINIT(onnxruntime),
	NULL,
	PHP_MINFO(onnxruntime),
	PHP_ONNXRUNTIME_VERSION,
	STANDARD_MODULE_PROPERTIES
};

#ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
#endif

#ifdef COMPILE_DL_ONNXRUNTIME
ZEND_GET_MODULE(onnxruntime)
#endif
