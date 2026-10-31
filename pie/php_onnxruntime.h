#ifndef PHP_ONNXRUNTIME_H
#define PHP_ONNXRUNTIME_H

#include "php.h"
#include "onnxruntime_c_api.h"

#if __has_include("version.h")
#include "version.h"
#endif
#ifndef PHP_ONNXRUNTIME_VERSION
#define PHP_ONNXRUNTIME_VERSION "0.1.0"
#endif
#define ONNX_RANK_MAX 8

extern const OrtApi *ort_api;
extern OrtEnv *ort_env;
extern OrtMemoryInfo *ort_cpu_mem;
extern OrtAllocator *ort_allocator;
extern zend_class_entry *onnx_exception_ce;
extern zend_class_entry *onnx_value_ce;
extern zend_class_entry *onnx_options_ce;

extern zend_module_entry onnxruntime_module_entry;
#define phpext_onnxruntime_ptr &onnxruntime_module_entry

int onnx_ort_ok(OrtStatus *status);
void onnx_throw(const char *message);
void onnx_throw_feed(const char *message, const char *input, zval *expected, zval *actual);

const char *onnx_dtype_name(ONNXTensorElementDataType type);
int onnx_dtype_by_name(const char *name, size_t len, ONNXTensorElementDataType *type);
size_t onnx_dtype_size(ONNXTensorElementDataType type);

#define ONNX_KIND_TENSOR 0
#define ONNX_KIND_SEQUENCE 1
#define ONNX_KIND_MAP 2

typedef struct _onnx_value {
	uint8_t kind;
	zend_bool from_bytes;
	ONNXTensorElementDataType dtype;
	int64_t *dims;
	size_t rank;
	void *data;
	size_t nbytes;
	zend_string *bytes;
	zend_string **strings;
	size_t nstrings;
	zval items;
	zval entries;
	zval map_keys;
	zval map_values;
	zend_object std;
} onnx_value;

static inline onnx_value *onnx_value_from_obj(zend_object *obj)
{
	return (onnx_value *) ((char *) obj - XtOffsetOf(onnx_value, std));
}

onnx_value *onnx_value_create(zval *zv);
void onnx_register_values(void);
int onnx_value_from_array(zval *values, const char *type_name, int use_model_type, ONNXTensorElementDataType model_type, zval *out, const char **error);
int onnx_value_from_ort(OrtValue *src, zval *out);
int onnx_value_to_ort(onnx_value *value, OrtValue **out, const char **error);

typedef struct _onnx_options {
	zend_long thread_count;
	zend_bool thread_count_set;
	zend_long parallel_thread_count;
	zend_bool parallel_thread_count_set;
	zval optimization;
	zend_bool parallel;
	zend_bool parallel_set;
	zval log_level;
	zend_long verbosity;
	zend_bool verbosity_set;
	zval free_dimensions;
	zval providers;
	zend_object std;
} onnx_options;

static inline onnx_options *onnx_options_from_obj(zend_object *obj)
{
	return (onnx_options *) ((char *) obj - XtOffsetOf(onnx_options, std));
}

void onnx_register_sessions(void);
zend_string *onnx_provider_short_name(const char *id);

#endif
