#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "zend_exceptions.h"
#include "zend_smart_str.h"
#include "zend_enum.h"
#include "php_onnxruntime.h"

#include <string.h>
#ifdef PHP_WIN32
#include <windows.h>
#ifndef strcasecmp
#define strcasecmp _stricmp
#endif
#define onnx_lock_init(lock) InitializeSRWLock(lock)
#define onnx_lock_destroy(lock) ((void) (lock))
#define onnx_lock_lock(lock) AcquireSRWLockExclusive(lock)
#define onnx_lock_unlock(lock) ReleaseSRWLockExclusive(lock)
#else
#include <strings.h>
#include <pthread.h>
#define onnx_lock_init(lock) pthread_mutex_init(lock, NULL)
#define onnx_lock_destroy(lock) pthread_mutex_destroy(lock)
#define onnx_lock_lock(lock) pthread_mutex_lock(lock)
#define onnx_lock_unlock(lock) pthread_mutex_unlock(lock)
#endif

static zend_class_entry *onnx_session_ce = NULL;
static zend_class_entry *onnx_value_info_ce = NULL;
static zend_class_entry *onnx_metadata_ce = NULL;
static zend_object_handlers onnx_session_handlers;
static zend_object_handlers onnx_value_info_handlers;
static zend_object_handlers onnx_metadata_handlers;

typedef struct _ones_buf {
	ONNXTensorElementDataType dtype;
	size_t rank;
	int64_t dims[ONNX_RANK_MAX];
	void *data;
	size_t nbytes;
} ones_buf;

typedef struct _onnx_session {
	OrtSession *session;
	zval providers;
	ones_buf *ones;
	size_t ones_count;
	size_t ones_size;
#ifdef PHP_WIN32
	SRWLOCK ones_lock;
#else
	pthread_mutex_t ones_lock;
#endif
	zend_object std;
} onnx_session;

typedef struct _onnx_value_info_obj {
	zend_string *name;
	zend_string *kind;
	zend_string *type;
	zval shape;
	zend_object std;
} onnx_value_info_obj;

typedef struct _onnx_metadata_obj {
	zend_string *producer;
	zend_string *graph_name;
	zend_string *domain;
	zend_string *description;
	zend_long version;
	zval custom;
	zend_object std;
} onnx_metadata_obj;

typedef struct _onnx_io {
	char *name;
	int optional;
	int initializer;
	int is_tensor;
	char *kind;
	char *decl;
	ONNXTensorElementDataType dtype;
	int64_t *dims;
	char **names;
	size_t rank;
} onnx_io;

static inline onnx_session *session_from_obj(zend_object *object)
{
	return (onnx_session *) ((char *) object - XtOffsetOf(onnx_session, std));
}

static inline onnx_value_info_obj *info_from_obj(zend_object *object)
{
	return (onnx_value_info_obj *) ((char *) object - XtOffsetOf(onnx_value_info_obj, std));
}

static inline onnx_metadata_obj *metadata_from_obj(zend_object *object)
{
	return (onnx_metadata_obj *) ((char *) object - XtOffsetOf(onnx_metadata_obj, std));
}

static void io_free(onnx_io *io)
{
	size_t i;

	if (io->name != NULL) {
		efree(io->name);
	}
	if (io->kind != NULL) {
		efree(io->kind);
	}
	if (io->decl != NULL) {
		efree(io->decl);
	}
	if (io->dims != NULL) {
		efree(io->dims);
	}
	if (io->names != NULL) {
		for (i = 0; i < io->rank; i++) {
			if (io->names[i] != NULL) {
				efree(io->names[i]);
			}
		}
		efree(io->names);
	}
	memset(io, 0, sizeof(*io));
}

static char *dup_cstr(const char *value)
{
	size_t length = value != NULL ? strlen(value) : 0;
	char *copy = emalloc(length + 1);

	if (length > 0) {
		memcpy(copy, value, length);
	}
	copy[length] = '\0';
	return copy;
}

static const char *enum_value(zval *enum_value)
{
	zval *backing = zend_enum_fetch_case_value(Z_OBJ_P(enum_value));

	return Z_TYPE_P(backing) == IS_STRING ? Z_STRVAL_P(backing) : "";
}

static const char *provider_alias(const char *name)
{
	size_t length = strlen(name);
	const char *suffix = "ExecutionProvider";
	size_t suffix_len = strlen(suffix);

	if (strcasecmp(name, "cpu") == 0) {
		return "CPUExecutionProvider";
	}
	if (strcasecmp(name, "cuda") == 0) {
		return "CUDAExecutionProvider";
	}
	if (strcasecmp(name, "coreml") == 0) {
		return "CoreMLExecutionProvider";
	}
	if (strcasecmp(name, "tensorrt") == 0) {
		return "TensorrtExecutionProvider";
	}
	if (length > suffix_len && memcmp(name + length - suffix_len, suffix, suffix_len) == 0) {
		return name;
	}
	return NULL;
}

/* A short name from this build, such as webgpu, is not in the alias list.
 * Match it against GetAvailableProviders and return that build's id. */
static char *provider_id_from_build(const char *name)
{
	char **providers = NULL;
	int count = 0;
	int i;
	char *found = NULL;
	OrtStatus *status;

	status = ort_api->GetAvailableProviders(&providers, &count);
	if (status != NULL) {
		ort_api->ReleaseStatus(status);
		return NULL;
	}
	for (i = 0; i < count; i++) {
		zend_string *short_name = onnx_provider_short_name(providers[i]);

		if (strcasecmp(ZSTR_VAL(short_name), name) == 0) {
			found = estrdup(providers[i]);
			zend_string_release(short_name);
			break;
		}
		zend_string_release(short_name);
	}
	status = ort_api->ReleaseAvailableProviders(providers, count);
	if (status != NULL) {
		ort_api->ReleaseStatus(status);
	}
	return found;
}

static char *resolve_provider_id(const char *name)
{
	const char *alias = provider_alias(name);

	if (alias != NULL) {
		return estrdup(alias);
	}
	return provider_id_from_build(name);
}

static int append_provider(OrtSessionOptions *options, const char *name, zval *provider_options, zval *shorts)
{
	char *id = resolve_provider_id(name);
	const char **keys = NULL;
	const char **values = NULL;
	size_t count = 0;
	zend_string *key;
	zval *value;
	OrtStatus *status;
	char *message;
	zend_string *short_name;

	if (id == NULL) {
		spprintf(&message, 0, "unknown provider '%s'", name);
		onnx_throw(message);
		efree(message);
		return 0;
	}
	if (provider_options != NULL) {
		count = zend_hash_num_elements(Z_ARRVAL_P(provider_options));
		keys = ecalloc(count > 0 ? count : 1, sizeof(char *));
		values = ecalloc(count > 0 ? count : 1, sizeof(char *));
		count = 0;
		ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL_P(provider_options), key, value) {
			if (key == NULL || Z_TYPE_P(value) != IS_STRING) {
				efree(keys);
				efree(values);
				efree(id);
				onnx_throw("provider options must be a map of strings");
				return 0;
			}
			keys[count] = ZSTR_VAL(key);
			values[count] = Z_STRVAL_P(value);
			count++;
		} ZEND_HASH_FOREACH_END();
	}
	status = ort_api->SessionOptionsAppendExecutionProvider(options, id, keys, values, count);
	if (keys != NULL) {
		efree(keys);
		efree(values);
	}
	if (status != NULL) {
		short_name = onnx_provider_short_name(id);
		spprintf(&message, 0, "provider %s: %s", ZSTR_VAL(short_name), ort_api->GetErrorMessage(status));
		ort_api->ReleaseStatus(status);
		zend_string_release(short_name);
		efree(id);
		onnx_throw(message);
		efree(message);
		return 0;
	}
	add_next_index_str(shorts, onnx_provider_short_name(id));
	efree(id);
	return 1;
}

static int configure_session(OrtSessionOptions *options, zval *options_zv, zval *shorts)
{
	onnx_options *settings = NULL;
	GraphOptimizationLevel optimization = ORT_ENABLE_ALL;
	int log_level = ORT_LOGGING_LEVEL_WARNING;
	zval *entry;

	array_init(shorts);
	if (options_zv != NULL) {
		settings = onnx_options_from_obj(Z_OBJ_P(options_zv));
		if (settings->thread_count_set && settings->thread_count < 0) {
			onnx_throw("intra-op thread count must not be negative");
			goto fail;
		}
		if (settings->parallel_thread_count_set && settings->parallel_thread_count < 0) {
			onnx_throw("inter-op thread count must not be negative");
			goto fail;
		}
		if (settings->verbosity_set && settings->verbosity < 0) {
			onnx_throw("verbosity must not be negative");
			goto fail;
		}
	}
	if (settings != NULL && settings->thread_count_set && settings->thread_count > 0) {
		if (!onnx_ort_ok(ort_api->SetIntraOpNumThreads(options, (int) settings->thread_count))) {
			goto fail;
		}
	}
	if (settings != NULL && settings->parallel_set) {
		if (!onnx_ort_ok(ort_api->SetSessionExecutionMode(options, settings->parallel ? ORT_PARALLEL : ORT_SEQUENTIAL))) {
			goto fail;
		}
		if (settings->parallel && settings->parallel_thread_count_set && settings->parallel_thread_count > 0) {
			if (!onnx_ort_ok(ort_api->SetInterOpNumThreads(options, (int) settings->parallel_thread_count))) {
				goto fail;
			}
		}
	}
	if (settings != NULL && Z_TYPE(settings->optimization) == IS_OBJECT) {
		const char *name = enum_value(&settings->optimization);
		if (strcmp(name, "off") == 0) {
			optimization = ORT_DISABLE_ALL;
		} else if (strcmp(name, "basic") == 0) {
			optimization = ORT_ENABLE_BASIC;
		} else if (strcmp(name, "extended") == 0) {
			optimization = ORT_ENABLE_EXTENDED;
		} else {
			optimization = ORT_ENABLE_ALL;
		}
	}
	if (!onnx_ort_ok(ort_api->SetSessionGraphOptimizationLevel(options, optimization))) {
		goto fail;
	}
	if (settings != NULL && Z_TYPE(settings->log_level) == IS_OBJECT) {
		const char *name = enum_value(&settings->log_level);
		if (strcmp(name, "verbose") == 0) {
			log_level = ORT_LOGGING_LEVEL_VERBOSE;
		} else if (strcmp(name, "info") == 0) {
			log_level = ORT_LOGGING_LEVEL_INFO;
		} else if (strcmp(name, "error") == 0) {
			log_level = ORT_LOGGING_LEVEL_ERROR;
		} else if (strcmp(name, "fatal") == 0) {
			log_level = ORT_LOGGING_LEVEL_FATAL;
		} else {
			log_level = ORT_LOGGING_LEVEL_WARNING;
		}
	}
	if (!onnx_ort_ok(ort_api->SetSessionLogSeverityLevel(options, log_level))) {
		goto fail;
	}
	if (settings != NULL && settings->verbosity_set) {
		if (!onnx_ort_ok(ort_api->SetSessionLogVerbosityLevel(options, (int) settings->verbosity))) {
			goto fail;
		}
	}
	if (settings != NULL && Z_TYPE(settings->free_dimensions) == IS_ARRAY) {
		zend_string *name;
		zval *size;

		ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL(settings->free_dimensions), name, size) {
			zval *entry = size;

			if (name == NULL) {
				continue;
			}
			ZVAL_DEREF(entry);
			if (!onnx_ort_ok(ort_api->AddFreeDimensionOverrideByName(options, ZSTR_VAL(name), (int64_t) zval_get_long(entry)))) {
				goto fail;
			}
		} ZEND_HASH_FOREACH_END();
	}
	if (settings == NULL || Z_TYPE(settings->providers) == IS_UNDEF) {
		if (!append_provider(options, "cpu", NULL, shorts)) {
			goto fail;
		}
		return 1;
	}
	if (!zend_array_is_list(Z_ARRVAL(settings->providers))) {
		onnx_throw("providers must be a list");
		goto fail;
	}
	if (zend_hash_num_elements(Z_ARRVAL(settings->providers)) == 0) {
		onnx_throw("provider list is empty");
		goto fail;
	}
	ZEND_HASH_FOREACH_VAL(Z_ARRVAL(settings->providers), entry) {
		ZVAL_DEREF(entry);
		if (Z_TYPE_P(entry) == IS_STRING) {
			if (!append_provider(options, Z_STRVAL_P(entry), NULL, shorts)) {
				goto fail;
			}
		} else if (Z_TYPE_P(entry) == IS_ARRAY) {
			zval *name;
			zval *provider_options;
			if (!zend_array_is_list(Z_ARRVAL_P(entry)) || zend_hash_num_elements(Z_ARRVAL_P(entry)) != 2) {
				onnx_throw("a provider entry must be a name or [name, options]");
				goto fail;
			}
			name = zend_hash_index_find(Z_ARRVAL_P(entry), 0);
			provider_options = zend_hash_index_find(Z_ARRVAL_P(entry), 1);
			if (name == NULL || Z_TYPE_P(name) != IS_STRING || provider_options == NULL || Z_TYPE_P(provider_options) != IS_ARRAY) {
				onnx_throw("a provider entry must be a name or [name, options]");
				goto fail;
			}
			if (!append_provider(options, Z_STRVAL_P(name), provider_options, shorts)) {
				goto fail;
			}
		} else {
			onnx_throw("a provider entry must be a name or [name, options]");
			goto fail;
		}
	} ZEND_HASH_FOREACH_END();
	return 1;

fail:
	zval_ptr_dtor(shorts);
	ZVAL_UNDEF(shorts);
	return 0;
}

static int open_session(onnx_session *session, const char *path, const void *bytes, size_t length, zval *options_zv)
{
	OrtSessionOptions *options = NULL;
	zval providers;
	OrtStatus *status;
	char *message;

	ZVAL_UNDEF(&providers);
	if (!onnx_ort_ok(ort_api->CreateSessionOptions(&options))) {
		return 0;
	}
	if (!configure_session(options, options_zv, &providers)) {
		ort_api->ReleaseSessionOptions(options);
		return 0;
	}
	if (path != NULL) {
		status = ort_api->CreateSession(ort_env, path, options, &session->session);
	} else {
		status = ort_api->CreateSessionFromArray(ort_env, bytes, length, options, &session->session);
	}
	ort_api->ReleaseSessionOptions(options);
	if (status != NULL) {
		if (path != NULL) {
			spprintf(&message, 0, "failed to load %s: %s", path, ort_api->GetErrorMessage(status));
		} else {
			spprintf(&message, 0, "failed to load model contents: %s", ort_api->GetErrorMessage(status));
		}
		ort_api->ReleaseStatus(status);
		onnx_throw(message);
		efree(message);
		zval_ptr_dtor(&providers);
		return 0;
	}
	ZVAL_COPY_VALUE(&session->providers, &providers);
	return 1;
}

static int describe_tensor(const OrtTensorTypeAndShapeInfo *info, onnx_io *io)
{
	size_t rank = 0;
	const char **symbols = NULL;
	size_t i;

	if (!onnx_ort_ok(ort_api->GetTensorElementType(info, &io->dtype))) {
		return 0;
	}
	if (!onnx_ort_ok(ort_api->GetDimensionsCount(info, &rank))) {
		return 0;
	}
	io->is_tensor = 1;
	io->kind = dup_cstr("tensor");
	io->decl = dup_cstr(onnx_dtype_name(io->dtype));
	io->rank = rank;
	if (rank == 0) {
		return 1;
	}
	io->dims = emalloc(sizeof(int64_t) * rank);
	io->names = ecalloc(rank, sizeof(char *));
	symbols = ecalloc(rank, sizeof(char *));
	if (!onnx_ort_ok(ort_api->GetDimensions(info, io->dims, rank)) || !onnx_ort_ok(ort_api->GetSymbolicDimensions(info, symbols, rank))) {
		efree(symbols);
		return 0;
	}
	for (i = 0; i < rank; i++) {
		if (io->dims[i] < 0 && symbols[i] != NULL && symbols[i][0] != '\0') {
			io->names[i] = dup_cstr(symbols[i]);
		}
	}
	efree(symbols);
	return 1;
}

static int describe_type(const OrtTypeInfo *type_info, onnx_io *io);

static int describe_contained(const OrtTypeInfo *type_info, char **decl, char **kind, int *is_tensor, ONNXTensorElementDataType *dtype, int64_t **dims, char ***names, size_t *rank)
{
	onnx_io inner;

	memset(&inner, 0, sizeof(inner));
	if (!describe_type(type_info, &inner)) {
		io_free(&inner);
		return 0;
	}
	*decl = inner.decl;
	*kind = inner.kind;
	*is_tensor = inner.is_tensor;
	*dtype = inner.dtype;
	*dims = inner.dims;
	*names = inner.names;
	*rank = inner.rank;
	inner.decl = NULL;
	inner.kind = NULL;
	inner.dims = NULL;
	inner.names = NULL;
	io_free(&inner);
	return 1;
}

static int describe_type(const OrtTypeInfo *type_info, onnx_io *io)
{
	ONNXType onnx_type;

	if (!onnx_ort_ok(ort_api->GetOnnxTypeFromTypeInfo(type_info, &onnx_type))) {
		return 0;
	}
	if (onnx_type == ONNX_TYPE_OPTIONAL) {
		const OrtOptionalTypeInfo *optional = NULL;
		OrtTypeInfo *contained = NULL;
		int ok;

		io->optional = 1;
		if (!onnx_ort_ok(ort_api->CastTypeInfoToOptionalTypeInfo(type_info, &optional))) {
			return 0;
		}
		if (!onnx_ort_ok(ort_api->GetOptionalContainedTypeInfo(optional, &contained))) {
			return 0;
		}
		ok = describe_type(contained, io);
		ort_api->ReleaseTypeInfo(contained);
		return ok;
	}
	if (onnx_type == ONNX_TYPE_TENSOR) {
		const OrtTensorTypeAndShapeInfo *tensor = NULL;
		if (!onnx_ort_ok(ort_api->CastTypeInfoToTensorInfo(type_info, &tensor)) || tensor == NULL) {
			onnx_throw("input type is not a tensor");
			return 0;
		}
		return describe_tensor(tensor, io);
	}
	if (onnx_type == ONNX_TYPE_SEQUENCE) {
		const OrtSequenceTypeInfo *sequence = NULL;
		OrtTypeInfo *element = NULL;
		char *decl = NULL;
		char *kind = NULL;
		int is_tensor = 0;
		ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		int64_t *dims = NULL;
		char **names = NULL;
		size_t rank = 0;

		if (!onnx_ort_ok(ort_api->CastTypeInfoToSequenceTypeInfo(type_info, &sequence)) || sequence == NULL) {
			return 0;
		}
		if (!onnx_ort_ok(ort_api->GetSequenceElementType(sequence, &element))) {
			return 0;
		}
		if (!describe_contained(element, &decl, &kind, &is_tensor, &dtype, &dims, &names, &rank)) {
			ort_api->ReleaseTypeInfo(element);
			return 0;
		}
		ort_api->ReleaseTypeInfo(element);
		io->kind = dup_cstr("sequence");
		spprintf(&io->decl, 0, "sequence<%s>", decl);
		efree(decl);
		efree(kind);
		if (dims != NULL) {
			efree(dims);
		}
		if (names != NULL) {
			size_t i;
			for (i = 0; i < rank; i++) {
				if (names[i] != NULL) {
					efree(names[i]);
				}
			}
			efree(names);
		}
		return 1;
	}
	if (onnx_type == ONNX_TYPE_MAP) {
		const OrtMapTypeInfo *map = NULL;
		OrtTypeInfo *value_type = NULL;
		ONNXTensorElementDataType key_type;
		char *decl = NULL;
		char *kind = NULL;
		int is_tensor = 0;
		ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		int64_t *dims = NULL;
		char **names = NULL;
		size_t rank = 0;

		if (!onnx_ort_ok(ort_api->CastTypeInfoToMapTypeInfo(type_info, &map)) || map == NULL) {
			return 0;
		}
		if (!onnx_ort_ok(ort_api->GetMapKeyType(map, &key_type))) {
			return 0;
		}
		if (!onnx_ort_ok(ort_api->GetMapValueType(map, &value_type))) {
			return 0;
		}
		if (!describe_contained(value_type, &decl, &kind, &is_tensor, &dtype, &dims, &names, &rank)) {
			ort_api->ReleaseTypeInfo(value_type);
			return 0;
		}
		ort_api->ReleaseTypeInfo(value_type);
		io->kind = dup_cstr("map");
		spprintf(&io->decl, 0, "map<%s,%s>", onnx_dtype_name(key_type), decl);
		efree(decl);
		efree(kind);
		if (dims != NULL) {
			efree(dims);
		}
		if (names != NULL) {
			size_t i;
			for (i = 0; i < rank; i++) {
				if (names[i] != NULL) {
					efree(names[i]);
				}
			}
			efree(names);
		}
		return 1;
	}
	onnx_throw("unsupported model type");
	return 0;
}

static int load_io(OrtSession *session, int outputs, onnx_io **out, size_t *count)
{
	size_t n = 0;
	size_t i;
	onnx_io *items;
	size_t initializers = 0;

	if (outputs) {
		if (!onnx_ort_ok(ort_api->SessionGetOutputCount(session, &n))) {
			return 0;
		}
	} else if (!onnx_ort_ok(ort_api->SessionGetInputCount(session, &n))) {
		return 0;
	}
	items = ecalloc(n > 0 ? n : 1, sizeof(onnx_io));
	for (i = 0; i < n; i++) {
		char *name = NULL;
		OrtTypeInfo *type_info = NULL;
		OrtStatus *status;

		if (outputs) {
			status = ort_api->SessionGetOutputName(session, i, ort_allocator, &name);
		} else {
			status = ort_api->SessionGetInputName(session, i, ort_allocator, &name);
		}
		if (!onnx_ort_ok(status)) {
			goto fail;
		}
		items[i].name = dup_cstr(name);
		ort_allocator->Free(ort_allocator, name);
		if (outputs) {
			status = ort_api->SessionGetOutputTypeInfo(session, i, &type_info);
		} else {
			status = ort_api->SessionGetInputTypeInfo(session, i, &type_info);
		}
		if (!onnx_ort_ok(status)) {
			goto fail;
		}
		if (!describe_type(type_info, &items[i])) {
			ort_api->ReleaseTypeInfo(type_info);
			goto fail;
		}
		ort_api->ReleaseTypeInfo(type_info);
	}
	if (!outputs && onnx_ort_ok(ort_api->SessionGetOverridableInitializerCount(session, &initializers))) {
		for (i = 0; i < initializers; i++) {
			char *name = NULL;
			size_t j;

			if (!onnx_ort_ok(ort_api->SessionGetOverridableInitializerName(session, i, ort_allocator, &name))) {
				goto fail;
			}
			for (j = 0; j < n; j++) {
				if (strcmp(items[j].name, name) == 0) {
					items[j].initializer = 1;
				}
			}
			ort_allocator->Free(ort_allocator, name);
		}
	} else if (!outputs && EG(exception)) {
		goto fail;
	}
	*out = items;
	*count = n;
	return 1;

fail:
	for (i = 0; i < n; i++) {
		io_free(&items[i]);
	}
	efree(items);
	return 0;
}

static void model_shape_zval(onnx_io *io, zval *out)
{
	size_t i;

	array_init_size(out, io->rank);
	for (i = 0; i < io->rank; i++) {
		if (io->dims != NULL && io->dims[i] >= 0) {
			add_next_index_long(out, io->dims[i]);
		} else if (io->names != NULL && io->names[i] != NULL) {
			add_next_index_string(out, io->names[i]);
		} else {
			add_next_index_null(out);
		}
	}
}

static void value_shape_zval(onnx_value *value, zval *out)
{
	size_t i;

	array_init_size(out, value->rank);
	for (i = 0; i < value->rank; i++) {
		add_next_index_long(out, value->dims[i]);
	}
}

static char *shape_text_from_zval(zval *shape)
{
	smart_str buffer = {0};
	zval *element;
	int first = 1;
	char *text;

	smart_str_appendc(&buffer, '[');
	if (Z_TYPE_P(shape) == IS_ARRAY) {
		ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape), element) {
			if (!first) {
				smart_str_appends(&buffer, ", ");
			}
			first = 0;
			if (Z_TYPE_P(element) == IS_LONG) {
				smart_str_append_long(&buffer, Z_LVAL_P(element));
			} else if (Z_TYPE_P(element) == IS_STRING) {
				smart_str_append(&buffer, Z_STR_P(element));
			} else {
				smart_str_appends(&buffer, "null");
			}
		} ZEND_HASH_FOREACH_END();
	}
	smart_str_appendc(&buffer, ']');
	smart_str_0(&buffer);
	text = estrdup(buffer.s != NULL ? ZSTR_VAL(buffer.s) : "[]");
	smart_str_free(&buffer);
	return text;
}

static void throw_mismatch(onnx_io *io, const char *got_type, zval *actual)
{
	zval expected;
	char *expected_text;
	char *actual_text;
	char *message;

	model_shape_zval(io, &expected);
	expected_text = shape_text_from_zval(&expected);
	actual_text = actual != NULL ? shape_text_from_zval(actual) : NULL;
	if (got_type != NULL && actual_text != NULL) {
		spprintf(&message, 0, "input '%s': expected %s shape %s, got %s shape %s", io->name, io->decl, expected_text, got_type, actual_text);
	} else if (got_type != NULL) {
		spprintf(&message, 0, "input '%s': expected %s shape %s, got %s", io->name, io->decl, expected_text, got_type);
	} else {
		spprintf(&message, 0, "input '%s': expected shape %s", io->name, expected_text);
	}
	onnx_throw_feed(message, io->name, &expected, actual);
	efree(message);
	efree(expected_text);
	if (actual_text != NULL) {
		efree(actual_text);
	}
	zval_ptr_dtor(&expected);
}

static int align_shape(onnx_io *io, const int64_t *value_dims, size_t value_rank, int64_t *out)
{
	int64_t promoted[ONNX_RANK_MAX];
	const int64_t *dims = value_dims;
	size_t rank = value_rank;
	size_t i;

	if (value_rank + 1 == io->rank && io->dims != NULL && io->dims[0] < 0) {
		promoted[0] = 1;
		if (value_rank > 0) {
			memcpy(promoted + 1, value_dims, sizeof(int64_t) * value_rank);
		}
		dims = promoted;
		rank = io->rank;
	}
	if (rank != io->rank) {
		return 0;
	}
	for (i = 0; i < rank; i++) {
		if (io->dims[i] >= 0 && io->dims[i] != dims[i]) {
			return 0;
		}
		out[i] = dims[i];
	}
	return 1;
}

static int bind_packed(onnx_io *io, const void *data, size_t length, OrtValue **out)
{
	int64_t concrete[ONNX_RANK_MAX];
	size_t element_size = onnx_dtype_size(io->dtype);
	size_t known = 1;
	int dynamic = 0;
	int dynamic_at = -1;
	size_t i;
	char *message;
	zval expected;

	if (!io->is_tensor || element_size == 0) {
		model_shape_zval(io, &expected);
		spprintf(&message, 0, "input '%s': this element type must be built with OnnxValue::tensor()", io->name);
		onnx_throw_feed(message, io->name, &expected, NULL);
		efree(message);
		zval_ptr_dtor(&expected);
		return 0;
	}
	for (i = 0; i < io->rank; i++) {
		if (io->dims[i] < 0) {
			dynamic++;
			dynamic_at = (int) i;
		} else if (io->dims[i] > 0 && known > SIZE_MAX / (size_t) io->dims[i]) {
			onnx_throw("tensor shape is too large");
			return 0;
		} else {
			known *= (size_t) io->dims[i];
		}
	}
	if (dynamic >= 2) {
		model_shape_zval(io, &expected);
		spprintf(&message, 0, "input '%s': two or more dynamic dimensions cannot be solved from packed bytes", io->name);
		onnx_throw_feed(message, io->name, &expected, NULL);
		efree(message);
		zval_ptr_dtor(&expected);
		return 0;
	}
	if (dynamic == 0) {
		if (length != known * element_size) {
			char *text;
			model_shape_zval(io, &expected);
			text = shape_text_from_zval(&expected);
			spprintf(&message, 0, "input '%s': packed length %zu does not match shape %s", io->name, length, text);
			efree(text);
			onnx_throw_feed(message, io->name, &expected, NULL);
			efree(message);
			zval_ptr_dtor(&expected);
			return 0;
		}
		if (io->rank > 0) {
			memcpy(concrete, io->dims, sizeof(int64_t) * io->rank);
		}
	} else {
		size_t unit = known * element_size;
		size_t solved;
		if (unit == 0 || length % unit != 0) {
			char *text;
			model_shape_zval(io, &expected);
			text = shape_text_from_zval(&expected);
			spprintf(&message, 0, "input '%s': packed length %zu does not match shape %s", io->name, length, text);
			efree(text);
			onnx_throw_feed(message, io->name, &expected, NULL);
			efree(message);
			zval_ptr_dtor(&expected);
			return 0;
		}
		solved = length / unit;
		memcpy(concrete, io->dims, sizeof(int64_t) * io->rank);
		concrete[dynamic_at] = (int64_t) solved;
	}
	return onnx_ort_ok(ort_api->CreateTensorWithDataAsOrtValue(
		ort_cpu_mem,
		(void *) data,
		length,
		io->rank > 0 ? concrete : NULL,
		io->rank,
		io->dtype,
		out));
}

static int bind_value(onnx_io *io, onnx_value *value, OrtValue **out)
{
	int64_t concrete[ONNX_RANK_MAX];
	zval actual;
	const char *error = NULL;

	if (value->from_bytes) {
		if (io->is_tensor && io->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
			char *message;
			zval expected;
			model_shape_zval(io, &expected);
			spprintf(&message, 0, "input '%s': A PHP string is packed bytes. A string element belongs in a list or OnnxValue::tensor().", io->name);
			onnx_throw_feed(message, io->name, &expected, NULL);
			efree(message);
			zval_ptr_dtor(&expected);
			return 0;
		}
		return bind_packed(io, ZSTR_VAL(value->bytes), ZSTR_LEN(value->bytes), out);
	}
	if (value->kind == ONNX_KIND_TENSOR) {
		if (!io->is_tensor || value->dtype != io->dtype || !align_shape(io, value->dims, value->rank, concrete)) {
			value_shape_zval(value, &actual);
			throw_mismatch(io, onnx_dtype_name(value->dtype), &actual);
			zval_ptr_dtor(&actual);
			return 0;
		}
		if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
			onnx_value shaped = *value;
			int64_t *original = value->dims;
			shaped.dims = io->rank > 0 ? concrete : NULL;
			shaped.rank = io->rank;
			if (!onnx_value_to_ort(&shaped, out, &error)) {
				if (error != NULL && !EG(exception)) {
					char *message;
					spprintf(&message, 0, "input '%s': %s", io->name, error);
					onnx_throw_feed(message, io->name, NULL, NULL);
					efree(message);
				}
				return 0;
			}
			(void) original;
			return 1;
		}
		return onnx_ort_ok(ort_api->CreateTensorWithDataAsOrtValue(
			ort_cpu_mem,
			value->nbytes > 0 ? value->data : "",
			value->nbytes,
			io->rank > 0 ? concrete : NULL,
			io->rank,
			value->dtype,
			out));
	}
	if ((value->kind == ONNX_KIND_SEQUENCE && strcmp(io->kind, "sequence") == 0) || (value->kind == ONNX_KIND_MAP && strcmp(io->kind, "map") == 0)) {
		if (!onnx_value_to_ort(value, out, &error)) {
			if (error != NULL && !EG(exception)) {
				char *message;
				spprintf(&message, 0, "input '%s': %s", io->name, error);
				onnx_throw_feed(message, io->name, NULL, NULL);
				efree(message);
			}
			return 0;
		}
		return 1;
	}
	throw_mismatch(io, value->kind == ONNX_KIND_SEQUENCE ? "sequence" : "map", NULL);
	return 0;
}

typedef struct _mask_job {
	int kind;
	int left;
	int64_t *lengths;
	size_t nlengths;
	void *data;
} mask_job;

static int ends_with(const char *name, const char *suffix)
{
	size_t name_len = strlen(name);
	size_t suffix_len = strlen(suffix);

	return name_len >= suffix_len && memcmp(name + name_len - suffix_len, suffix, suffix_len) == 0;
}

static int mask_count(const int64_t *dims, size_t rank, size_t *count)
{
	size_t i;
	size_t total = 1;

	for (i = 0; i < rank; i++) {
		if (dims[i] < 0) {
			return 0;
		}
		if (dims[i] > 0 && total > SIZE_MAX / (size_t) dims[i]) {
			return 0;
		}
		total *= (size_t) dims[i];
	}
	*count = total;
	return 1;
}

static int store_flag(void *data, size_t index, ONNXTensorElementDataType dtype, int one)
{
	switch (dtype) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			((int64_t *) data)[index] = one ? 1 : 0;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			((int32_t *) data)[index] = one ? 1 : 0;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			((int8_t *) data)[index] = one ? 1 : 0;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			((uint8_t *) data)[index] = one ? 1 : 0;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			((float *) data)[index] = one ? 1.0f : 0.0f;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			((double *) data)[index] = one ? 1.0 : 0.0;
			return 1;
		default:
			return 0;
	}
}

static int lookup_dim(onnx_io *inputs, OrtValue **bound, size_t count, const char *symbol, int64_t *got)
{
	size_t i;
	size_t axis;
	int found = 0;

	for (i = 0; i < count; i++) {
		OrtTensorTypeAndShapeInfo *info = NULL;
		int64_t dims[ONNX_RANK_MAX];
		size_t rank = 0;
		int tensor = 0;

		if (bound[i] == NULL || inputs[i].names == NULL) {
			continue;
		}
		if (!onnx_ort_ok(ort_api->IsTensor(bound[i], &tensor)) || !tensor) {
			continue;
		}
		if (!onnx_ort_ok(ort_api->GetTensorTypeAndShape(bound[i], &info))) {
			return -1;
		}
		if (!onnx_ort_ok(ort_api->GetDimensionsCount(info, &rank)) || rank > ONNX_RANK_MAX
			|| !onnx_ort_ok(ort_api->GetDimensions(info, dims, rank))) {
			ort_api->ReleaseTensorTypeAndShapeInfo(info);
			return -1;
		}
		ort_api->ReleaseTensorTypeAndShapeInfo(info);
		for (axis = 0; axis < inputs[i].rank && axis < rank; axis++) {
			if (inputs[i].names[axis] != NULL && strcmp(inputs[i].names[axis], symbol) == 0) {
				if (found && *got != dims[axis]) {
					return -1;
				}
				*got = dims[axis];
				found = 1;
			}
		}
	}
	return found;
}

static int parse_sequence(onnx_io *io, zval *feed, mask_job *job)
{
	zend_string *key;
	zval *entry;
	zval *length;
	size_t count;
	size_t index = 0;
	char *message;

	ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL_P(feed), key, entry) {
		ZVAL_DEREF(entry);
		if (key == NULL) {
			spprintf(&message, 0, "input '%s': mask config key must be a name", io->name);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
		}
		if (strcmp(ZSTR_VAL(key), "sequence") == 0) {
			if (Z_TYPE_P(entry) != IS_ARRAY || !zend_array_is_list(Z_ARRVAL_P(entry))) {
				spprintf(&message, 0, "input '%s': sequence must be a list of lengths", io->name);
				onnx_throw_feed(message, io->name, NULL, NULL);
				efree(message);
				return 0;
			}
			count = zend_hash_num_elements(Z_ARRVAL_P(entry));
			job->lengths = ecalloc(count > 0 ? count : 1, sizeof(int64_t));
			job->nlengths = count;
			ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(entry), length) {
				ZVAL_DEREF(length);
				if (Z_TYPE_P(length) != IS_LONG || Z_LVAL_P(length) < 0) {
					spprintf(&message, 0, "input '%s': sequence must be a list of lengths", io->name);
					onnx_throw_feed(message, io->name, NULL, NULL);
					efree(message);
					return 0;
				}
				job->lengths[index++] = Z_LVAL_P(length);
			} ZEND_HASH_FOREACH_END();
		} else if (strcmp(ZSTR_VAL(key), "left") == 0) {
			if (Z_TYPE_P(entry) != IS_TRUE && Z_TYPE_P(entry) != IS_FALSE) {
				spprintf(&message, 0, "input '%s': left must be a bool", io->name);
				onnx_throw_feed(message, io->name, NULL, NULL);
				efree(message);
				return 0;
			}
			job->left = Z_TYPE_P(entry) == IS_TRUE;
		} else {
			spprintf(&message, 0, "input '%s': unknown mask option '%s'", io->name, ZSTR_VAL(key));
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
		}
	} ZEND_HASH_FOREACH_END();
	if (job->lengths == NULL) {
		spprintf(&message, 0, "input '%s': sequence must be a list of lengths", io->name);
		onnx_throw_feed(message, io->name, NULL, NULL);
		efree(message);
		return 0;
	}
	return 1;
}

static int classify_feed(onnx_io *io, zval *feed, mask_job *job)
{
	char *message;

	memset(job, 0, sizeof(*job));
	ZVAL_DEREF(feed);
	if (Z_TYPE_P(feed) == IS_LONG) {
		if (Z_LVAL_P(feed) == 1 && ends_with(io->name, "attention_mask")) {
			job->kind = 1;
			return 1;
		}
		if (ends_with(io->name, "attention_mask")) {
			spprintf(&message, 0, "input '%s': attention mask shorthand is 1", io->name);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
		}
		onnx_throw("a feed must be an OnnxValue, an array, a string, or the attention mask shorthand 1");
		return 0;
	}
	if (Z_TYPE_P(feed) == IS_ARRAY && !zend_array_is_list(Z_ARRVAL_P(feed))) {
		if (!ends_with(io->name, "attention_mask")) {
			spprintf(&message, 0, "input '%s': mask config is only for an attention_mask input", io->name);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
		}
		if (!parse_sequence(io, feed, job)) {
			return 0;
		}
		job->kind = 2;
		return 1;
	}
	return 1;
}

static int resolve_mask_shape(onnx_io *io, onnx_io *inputs, OrtValue **bound, size_t count, mask_job *job, int64_t *shape)
{
	size_t axis;
	char *message;

	for (axis = 0; axis < io->rank; axis++) {
		if (io->dims[axis] >= 0) {
			shape[axis] = io->dims[axis];
			continue;
		}
		if (io->names[axis] == NULL) {
			spprintf(&message, 0, "input '%s': cannot infer attention mask shape", io->name);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
		}
		int looked = lookup_dim(inputs, bound, count, io->names[axis], &shape[axis]);
		if (looked < 0) {
			spprintf(&message, 0, "input '%s': cannot infer attention mask shape", io->name);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
		}
		if (looked == 1) {
			continue;
		}
			if (job->kind == 2 && io->rank == 2 && axis == 0) {
				shape[axis] = (int64_t) job->nlengths;
				continue;
			}
			if (job->kind == 2 && io->rank == 2 && axis == 1) {
				size_t row;
				int64_t width = 0;
				for (row = 0; row < job->nlengths; row++) {
					if (job->lengths[row] > width) {
						width = job->lengths[row];
					}
				}
				shape[axis] = width;
				continue;
			}
			spprintf(&message, 0, "input '%s': cannot infer attention mask shape", io->name);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
	}
	return 1;
}

static void *ones_buffer(onnx_session *session, ONNXTensorElementDataType dtype, const int64_t *shape, size_t rank, size_t nbytes)
{
	size_t i;
	size_t index;
	void *data;
	ones_buf *grown;

	onnx_lock_lock(&session->ones_lock);
	for (i = 0; i < session->ones_count; i++) {
		if (session->ones[i].dtype == dtype && session->ones[i].rank == rank && session->ones[i].nbytes == nbytes
			&& memcmp(session->ones[i].dims, shape, sizeof(int64_t) * rank) == 0) {
			data = session->ones[i].data;
			onnx_lock_unlock(&session->ones_lock);
			return data;
		}
	}
	data = ecalloc(nbytes > 0 ? nbytes : 1, 1);
	for (index = 0; index < nbytes / onnx_dtype_size(dtype); index++) {
		if (!store_flag(data, index, dtype, 1)) {
			efree(data);
			onnx_lock_unlock(&session->ones_lock);
			return NULL;
		}
	}
	if (session->ones_count == session->ones_size) {
		session->ones_size = session->ones_size > 0 ? session->ones_size * 2 : 4;
		grown = erealloc(session->ones, sizeof(ones_buf) * session->ones_size);
		session->ones = grown;
	}
	session->ones[session->ones_count].dtype = dtype;
	session->ones[session->ones_count].rank = rank;
	memset(session->ones[session->ones_count].dims, 0, sizeof(int64_t) * ONNX_RANK_MAX);
	if (rank > 0) {
		memcpy(session->ones[session->ones_count].dims, shape, sizeof(int64_t) * rank);
	}
	session->ones[session->ones_count].data = data;
	session->ones[session->ones_count].nbytes = nbytes;
	session->ones_count++;
	onnx_lock_unlock(&session->ones_lock);
	return data;
}

static int build_mask(onnx_session *session, onnx_io *io, onnx_io *inputs, OrtValue **bound, size_t count, mask_job *job, OrtValue **out)
{
	int64_t shape[ONNX_RANK_MAX];
	size_t total = 0;
	size_t element_size;
	char *message;
	void *filled = NULL;

	if (!io->is_tensor || io->rank > ONNX_RANK_MAX) {
		spprintf(&message, 0, "input '%s': cannot infer attention mask shape", io->name);
		onnx_throw_feed(message, io->name, NULL, NULL);
		efree(message);
		return 0;
	}
	element_size = onnx_dtype_size(io->dtype);
	if (element_size == 0 || !resolve_mask_shape(io, inputs, bound, count, job, shape) || !mask_count(shape, io->rank, &total)) {
		if (!EG(exception)) {
			spprintf(&message, 0, "input '%s': cannot infer attention mask shape", io->name);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
		}
		return 0;
	}
	if (job->kind == 1 && total > 0) {
		filled = ones_buffer(session, io->dtype, shape, io->rank, total * element_size);
		if (filled == NULL) {
			spprintf(&message, 0, "input '%s': cannot fill this element type", io->name);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
		}
	} else {
		job->data = ecalloc(total > 0 ? total : 1, element_size);
		filled = job->data;
	}
	if (job->kind != 1) {
		int64_t batch = io->rank == 2 ? shape[0] : 1;
		int64_t width = io->rank == 2 ? shape[1] : shape[0];
		int64_t row;
		int64_t column;

		if (io->rank != 2 || (size_t) batch != job->nlengths) {
			spprintf(&message, 0, "input '%s': sequence has %zu rows but the batch is %lld", io->name, job->nlengths, (long long) batch);
			onnx_throw_feed(message, io->name, NULL, NULL);
			efree(message);
			return 0;
		}
		for (row = 0; row < batch; row++) {
			if (job->lengths[row] > width) {
				spprintf(&message, 0, "input '%s': sequence length %lld is longer than %lld", io->name, (long long) job->lengths[row], (long long) width);
				onnx_throw_feed(message, io->name, NULL, NULL);
				efree(message);
				return 0;
			}
			for (column = 0; column < width; column++) {
				int on = job->left ? column >= width - job->lengths[row] : column < job->lengths[row];
				if (!store_flag(job->data, (size_t) (row * width + column), io->dtype, on)) {
					spprintf(&message, 0, "input '%s': cannot fill this element type", io->name);
					onnx_throw_feed(message, io->name, NULL, NULL);
					efree(message);
					return 0;
				}
			}
		}
	}
	if (!onnx_ort_ok(ort_api->CreateTensorWithDataAsOrtValue(
		ort_cpu_mem,
		total > 0 ? filled : "",
		total * element_size,
		io->rank > 0 ? shape : NULL,
		io->rank,
		io->dtype,
		out))) {
		return 0;
	}
	return 1;
}

static int greater_at(void *base, ONNXTensorElementDataType dtype, size_t left, size_t right)
{
	switch (dtype) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			return ((float *) base)[left] > ((float *) base)[right];
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			return ((double *) base)[left] > ((double *) base)[right];
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			return ((int64_t *) base)[left] > ((int64_t *) base)[right];
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			return ((int32_t *) base)[left] > ((int32_t *) base)[right];
		default:
			return -1;
	}
}

static int parse_index_config(const char *name, zval *spec, int *mode)
{
	zend_string *key;
	zval *entry;
	int saw_index = 0;
	int each = 0;
	char *message;

	ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL_P(spec), key, entry) {
		ZVAL_DEREF(entry);
		if (key == NULL) {
			spprintf(&message, 0, "output '%s': index config key must be a name", name);
			onnx_throw(message);
			efree(message);
			return 0;
		}
		if (strcmp(ZSTR_VAL(key), "index") == 0) {
			if (Z_TYPE_P(entry) != IS_STRING || strcmp(Z_STRVAL_P(entry), "max") != 0) {
				spprintf(&message, 0, "output '%s': index must be 'max'", name);
				onnx_throw(message);
				efree(message);
				return 0;
			}
			saw_index = 1;
		} else if (strcmp(ZSTR_VAL(key), "step") == 0) {
			if (Z_TYPE_P(entry) != IS_STRING || strcmp(Z_STRVAL_P(entry), "each") != 0) {
				spprintf(&message, 0, "output '%s': step must be 'each'", name);
				onnx_throw(message);
				efree(message);
				return 0;
			}
			each = 1;
		} else {
			spprintf(&message, 0, "output '%s': unknown index option '%s'", name, ZSTR_VAL(key));
			onnx_throw(message);
			efree(message);
			return 0;
		}
	} ZEND_HASH_FOREACH_END();
	if (!saw_index) {
		spprintf(&message, 0, "output '%s': index must be 'max'", name);
		onnx_throw(message);
		efree(message);
		return 0;
	}
	*mode = each ? 2 : 1;
	return 1;
}

static int index_value(OrtValue *src, int each, const char *name, zval *out)
{
	OrtTensorTypeAndShapeInfo *info = NULL;
	ONNXTensorElementDataType dtype;
	int64_t dims[ONNX_RANK_MAX];
	int64_t result_dims[ONNX_RANK_MAX];
	size_t rank = 0;
	size_t result_rank;
	size_t vocab;
	size_t steps;
	size_t outer;
	size_t count;
	size_t item;
	void *base = NULL;
	int64_t *picked = NULL;
	OrtValue *reduced = NULL;
	int ok = 1;
	char *message;

	ZVAL_UNDEF(out);
	if (!onnx_ort_ok(ort_api->GetTensorTypeAndShape(src, &info))) {
		return 0;
	}
	if (!onnx_ort_ok(ort_api->GetTensorElementType(info, &dtype)) || !onnx_ort_ok(ort_api->GetDimensionsCount(info, &rank))
		|| rank > ONNX_RANK_MAX || !onnx_ort_ok(ort_api->GetDimensions(info, dims, rank))) {
		ort_api->ReleaseTensorTypeAndShapeInfo(info);
		return 0;
	}
	ort_api->ReleaseTensorTypeAndShapeInfo(info);
	if (rank == 0 || dims[rank - 1] <= 0 || (each && rank < 3)) {
		spprintf(&message, 0, each ? "output '%s': this output has no step dimension" : "output '%s': cannot index this shape", name);
		onnx_throw(message);
		efree(message);
		return 0;
	}
	if (!onnx_ort_ok(ort_api->GetTensorMutableData(src, &base))) {
		return 0;
	}
	vocab = (size_t) dims[rank - 1];
	if (each) {
		result_rank = rank - 1;
		memcpy(result_dims, dims, sizeof(int64_t) * result_rank);
		steps = (size_t) dims[rank - 2];
		outer = 1;
		for (item = 0; item + 2 < rank; item++) {
			outer *= (size_t) dims[item];
		}
	} else if (rank >= 3) {
		result_rank = rank - 2;
		if (result_rank > 0) {
			memcpy(result_dims, dims, sizeof(int64_t) * result_rank);
		}
		steps = (size_t) dims[rank - 2];
		outer = 1;
		for (item = 0; item + 2 < rank; item++) {
			outer *= (size_t) dims[item];
		}
		if (steps == 0) {
			spprintf(&message, 0, "output '%s': cannot index this shape", name);
			onnx_throw(message);
			efree(message);
			return 0;
		}
	} else if (rank == 2) {
		result_rank = 1;
		result_dims[0] = dims[0];
		steps = 1;
		outer = (size_t) dims[0];
		vocab = (size_t) dims[1];
	} else {
		result_rank = 0;
		steps = 1;
		outer = 1;
	}
	count = each ? outer * steps : outer;
	picked = ecalloc(count > 0 ? count : 1, sizeof(int64_t));
	for (item = 0; item < count; item++) {
		size_t step = each ? item % steps : (rank >= 3 ? steps - 1 : 0);
		size_t group = each ? item / steps : item;
		size_t start = (group * (rank >= 3 || each ? steps : 1) + step) * vocab;
		size_t cursor;
		int64_t best = 0;

		if (rank == 2 && !each) {
			start = item * vocab;
		}
		for (cursor = 1; cursor < vocab; cursor++) {
			int cmp = greater_at(base, dtype, start + cursor, start + best);
			if (cmp < 0) {
				ok = 0;
				break;
			}
			if (cmp) {
				best = (int64_t) cursor;
			}
		}
		if (!ok) {
			break;
		}
		picked[item] = best;
	}
	if (!ok) {
		efree(picked);
		spprintf(&message, 0, "output '%s': cannot index this element type", name);
		onnx_throw(message);
		efree(message);
		return 0;
	}
	if (!onnx_ort_ok(ort_api->CreateTensorWithDataAsOrtValue(
		ort_cpu_mem, count > 0 ? (void *) picked : "", count * sizeof(int64_t),
		result_rank > 0 ? result_dims : NULL, result_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &reduced))
		|| !onnx_value_from_ort(reduced, out)) {
		if (reduced != NULL) {
			ort_api->ReleaseValue(reduced);
		}
		efree(picked);
		return 0;
	}
	ort_api->ReleaseValue(reduced);
	efree(picked);
	return 1;
}

static int bind_feed(onnx_io *io, zval *feed, OrtValue **out, zval *temp)
{
	const char *error = NULL;
	char *message;
	zval expected;

	ZVAL_UNDEF(temp);
	ZVAL_DEREF(feed);
	if (Z_TYPE_P(feed) == IS_STRING) {
		if (io->is_tensor && io->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
			model_shape_zval(io, &expected);
			spprintf(&message, 0, "input '%s': A PHP string is packed bytes. A string element belongs in a list or OnnxValue::tensor().", io->name);
			onnx_throw_feed(message, io->name, &expected, NULL);
			efree(message);
			zval_ptr_dtor(&expected);
			return 0;
		}
		return bind_packed(io, Z_STRVAL_P(feed), Z_STRLEN_P(feed), out);
	}
	if (Z_TYPE_P(feed) == IS_ARRAY) {
		if (!io->is_tensor) {
			model_shape_zval(io, &expected);
			spprintf(&message, 0, "input '%s': expected %s", io->name, io->decl);
			onnx_throw_feed(message, io->name, &expected, NULL);
			efree(message);
			zval_ptr_dtor(&expected);
			return 0;
		}
		if (!onnx_value_from_array(feed, NULL, 1, io->dtype, temp, &error)) {
			model_shape_zval(io, &expected);
			spprintf(&message, 0, "input '%s': %s", io->name, error != NULL ? error : "tensor cannot be built");
			onnx_throw_feed(message, io->name, &expected, NULL);
			efree(message);
			zval_ptr_dtor(&expected);
			return 0;
		}
		if (!bind_value(io, onnx_value_from_obj(Z_OBJ_P(temp)), out)) {
			zval_ptr_dtor(temp);
			ZVAL_UNDEF(temp);
			return 0;
		}
		return 1;
	}
	if (Z_TYPE_P(feed) == IS_OBJECT && instanceof_function(Z_OBJCE_P(feed), onnx_value_ce)) {
		return bind_value(io, onnx_value_from_obj(Z_OBJ_P(feed)), out);
	}
	onnx_throw("a feed must be an OnnxValue, an array, or a string");
	return 0;
}

static zend_string *alloc_string(char *value)
{
	zend_string *result;

	if (value == NULL) {
		return ZSTR_EMPTY_ALLOC();
	}
	result = zend_string_init(value, strlen(value), 0);
	ort_allocator->Free(ort_allocator, value);
	return result;
}

static void make_info(zval *out, onnx_io *io)
{
	onnx_value_info_obj *info;

	object_init_ex(out, onnx_value_info_ce);
	info = info_from_obj(Z_OBJ_P(out));
	info->name = zend_string_init(io->name, strlen(io->name), 0);
	info->kind = zend_string_init(io->kind, strlen(io->kind), 0);
	info->type = zend_string_init(io->decl, strlen(io->decl), 0);
	model_shape_zval(io, &info->shape);
}

static zend_object *session_new(zend_class_entry *ce)
{
	onnx_session *session = zend_object_alloc(sizeof(onnx_session), ce);

	zend_object_std_init(&session->std, ce);
	object_properties_init(&session->std, ce);
	session->std.handlers = &onnx_session_handlers;
	session->session = NULL;
	session->ones = NULL;
	session->ones_count = 0;
	session->ones_size = 0;
	onnx_lock_init(&session->ones_lock);
	ZVAL_UNDEF(&session->providers);
	return &session->std;
}

static void release_session(onnx_session *session)
{
	if (session->session != NULL && ort_api != NULL) {
		ort_api->ReleaseSession(session->session);
		session->session = NULL;
	}
	if (session->ones != NULL) {
		size_t i;

		for (i = 0; i < session->ones_count; i++) {
			efree(session->ones[i].data);
		}
		efree(session->ones);
		session->ones = NULL;
		session->ones_count = 0;
		session->ones_size = 0;
	}
}

static int require_open(onnx_session *session, const char *method)
{
	char *message;

	if (session->session != NULL) {
		return 1;
	}
	spprintf(&message, 0, "OnnxSession::%s() cannot be called after close()", method);
	onnx_throw(message);
	efree(message);
	return 0;
}

static void session_free(zend_object *object)
{
	onnx_session *session = session_from_obj(object);

	release_session(session);
	onnx_lock_destroy(&session->ones_lock);
	if (Z_TYPE(session->providers) != IS_UNDEF) {
		zval_ptr_dtor(&session->providers);
	}
	zend_object_std_dtor(object);
}

static zend_object *info_new(zend_class_entry *ce)
{
	onnx_value_info_obj *info = zend_object_alloc(sizeof(onnx_value_info_obj), ce);

	zend_object_std_init(&info->std, ce);
	object_properties_init(&info->std, ce);
	info->std.handlers = &onnx_value_info_handlers;
	info->name = NULL;
	info->kind = NULL;
	info->type = NULL;
	ZVAL_UNDEF(&info->shape);
	return &info->std;
}

static void info_free(zend_object *object)
{
	onnx_value_info_obj *info = info_from_obj(object);

	if (info->name != NULL) {
		zend_string_release(info->name);
	}
	if (info->kind != NULL) {
		zend_string_release(info->kind);
	}
	if (info->type != NULL) {
		zend_string_release(info->type);
	}
	if (Z_TYPE(info->shape) != IS_UNDEF) {
		zval_ptr_dtor(&info->shape);
	}
	zend_object_std_dtor(object);
}

static zend_object *metadata_new(zend_class_entry *ce)
{
	onnx_metadata_obj *metadata = zend_object_alloc(sizeof(onnx_metadata_obj), ce);

	zend_object_std_init(&metadata->std, ce);
	object_properties_init(&metadata->std, ce);
	metadata->std.handlers = &onnx_metadata_handlers;
	metadata->producer = NULL;
	metadata->graph_name = NULL;
	metadata->domain = NULL;
	metadata->description = NULL;
	ZVAL_UNDEF(&metadata->custom);
	return &metadata->std;
}

static void metadata_free(zend_object *object)
{
	onnx_metadata_obj *metadata = metadata_from_obj(object);

	if (metadata->producer != NULL) {
		zend_string_release(metadata->producer);
	}
	if (metadata->graph_name != NULL) {
		zend_string_release(metadata->graph_name);
	}
	if (metadata->domain != NULL) {
		zend_string_release(metadata->domain);
	}
	if (metadata->description != NULL) {
		zend_string_release(metadata->description);
	}
	if (Z_TYPE(metadata->custom) != IS_UNDEF) {
		zval_ptr_dtor(&metadata->custom);
	}
	zend_object_std_dtor(object);
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_session_construct, 0, 0, 1)
	ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
	ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, options, Switon\\OnnxRuntime\\OnnxOptions, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_session_contents, 0, 1, IS_STATIC, 0)
	ZEND_ARG_TYPE_INFO(0, contents, IS_STRING, 0)
	ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, options, Switon\\OnnxRuntime\\OnnxOptions, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_session_list, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_session_void, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_session_metadata, 0, 0, Switon\\OnnxRuntime\\OnnxMetadata, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_session_run, 0, 1, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO(0, feeds, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, outputs, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_info_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_info_string, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_info_shape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_metadata_long, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

static PHP_METHOD(OnnxSession, __construct)
{
	char *path;
	size_t path_len;
	zval *options = NULL;

	ZEND_PARSE_PARAMETERS_START(1, 2)
		Z_PARAM_STRING(path, path_len)
		Z_PARAM_OPTIONAL
		Z_PARAM_OBJECT_OF_CLASS_OR_NULL(options, onnx_options_ce)
	ZEND_PARSE_PARAMETERS_END();
	(void) path_len;
	open_session(session_from_obj(Z_OBJ_P(ZEND_THIS)), path, NULL, 0, options);
}

static PHP_METHOD(OnnxSession, fromContents)
{
	zend_string *bytes;
	zval *options = NULL;
	zend_class_entry *ce = zend_get_called_scope(execute_data);

	ZEND_PARSE_PARAMETERS_START(1, 2)
		Z_PARAM_STR(bytes)
		Z_PARAM_OPTIONAL
		Z_PARAM_OBJECT_OF_CLASS_OR_NULL(options, onnx_options_ce)
	ZEND_PARSE_PARAMETERS_END();
	if (ce == NULL || !instanceof_function(ce, onnx_session_ce)) {
		ce = onnx_session_ce;
	}
	object_init_ex(return_value, ce);
	open_session(session_from_obj(Z_OBJ_P(return_value)), NULL, ZSTR_VAL(bytes), ZSTR_LEN(bytes), options);
}

static PHP_METHOD(OnnxSession, inputs)
{
	onnx_session *session = session_from_obj(Z_OBJ_P(ZEND_THIS));
	onnx_io *items = NULL;
	size_t count = 0;
	size_t i;

	ZEND_PARSE_PARAMETERS_NONE();
	if (!require_open(session, "inputs")) {
		return;
	}
	if (!load_io(session->session, 0, &items, &count)) {
		return;
	}
	array_init_size(return_value, count);
	for (i = 0; i < count; i++) {
		zval info;
		make_info(&info, &items[i]);
		add_next_index_zval(return_value, &info);
		io_free(&items[i]);
	}
	efree(items);
}

static PHP_METHOD(OnnxSession, outputs)
{
	onnx_session *session = session_from_obj(Z_OBJ_P(ZEND_THIS));
	onnx_io *items = NULL;
	size_t count = 0;
	size_t i;

	ZEND_PARSE_PARAMETERS_NONE();
	if (!require_open(session, "outputs")) {
		return;
	}
	if (!load_io(session->session, 1, &items, &count)) {
		return;
	}
	array_init_size(return_value, count);
	for (i = 0; i < count; i++) {
		zval info;
		make_info(&info, &items[i]);
		add_next_index_zval(return_value, &info);
		io_free(&items[i]);
	}
	efree(items);
}

static PHP_METHOD(OnnxSession, metadata)
{
	onnx_session *session = session_from_obj(Z_OBJ_P(ZEND_THIS));
	OrtModelMetadata *metadata = NULL;
	onnx_metadata_obj *object;
	char **keys = NULL;
	int64_t key_count = 0;
	int64_t i;
	char *value;

	ZEND_PARSE_PARAMETERS_NONE();
	if (!require_open(session, "metadata")) {
		return;
	}
	if (!onnx_ort_ok(ort_api->SessionGetModelMetadata(session->session, &metadata))) {
		return;
	}
	object_init_ex(return_value, onnx_metadata_ce);
	object = metadata_from_obj(Z_OBJ_P(return_value));
	if (!onnx_ort_ok(ort_api->ModelMetadataGetProducerName(metadata, ort_allocator, &value))) {
		goto fail;
	}
	object->producer = alloc_string(value);
	if (!onnx_ort_ok(ort_api->ModelMetadataGetGraphName(metadata, ort_allocator, &value))) {
		goto fail;
	}
	object->graph_name = alloc_string(value);
	if (!onnx_ort_ok(ort_api->ModelMetadataGetDomain(metadata, ort_allocator, &value))) {
		goto fail;
	}
	object->domain = alloc_string(value);
	if (!onnx_ort_ok(ort_api->ModelMetadataGetDescription(metadata, ort_allocator, &value))) {
		goto fail;
	}
	object->description = alloc_string(value);
	{
		int64_t version = 0;
		if (!onnx_ort_ok(ort_api->ModelMetadataGetVersion(metadata, &version))) {
			goto fail;
		}
		if (version == INT64_MAX) {
			version = 0;
		}
		object->version = (zend_long) version;
	}
	if (!onnx_ort_ok(ort_api->ModelMetadataGetCustomMetadataMapKeys(metadata, ort_allocator, &keys, &key_count))) {
		goto fail;
	}
	array_init(&object->custom);
	for (i = 0; i < key_count; i++) {
		char *custom = NULL;
		if (!onnx_ort_ok(ort_api->ModelMetadataLookupCustomMetadataMap(metadata, ort_allocator, keys[i], &custom))) {
			goto fail;
		}
		if (custom != NULL) {
			add_assoc_str(&object->custom, keys[i], alloc_string(custom));
		}
		ort_allocator->Free(ort_allocator, keys[i]);
		keys[i] = NULL;
	}
	if (keys != NULL) {
		ort_allocator->Free(ort_allocator, keys);
	}
	ort_api->ReleaseModelMetadata(metadata);
	return;

fail:
	if (keys != NULL) {
		for (i = 0; i < key_count; i++) {
			if (keys[i] != NULL) {
				ort_allocator->Free(ort_allocator, keys[i]);
			}
		}
		ort_allocator->Free(ort_allocator, keys);
	}
	ort_api->ReleaseModelMetadata(metadata);
}

static PHP_METHOD(OnnxSession, providers)
{
	onnx_session *session = session_from_obj(Z_OBJ_P(ZEND_THIS));

	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_COPY(&session->providers);
}

static PHP_METHOD(OnnxSession, close)
{
	onnx_session *session = session_from_obj(Z_OBJ_P(ZEND_THIS));

	ZEND_PARSE_PARAMETERS_NONE();
	release_session(session);
}

static PHP_METHOD(OnnxSession, run)
{
	onnx_session *session = session_from_obj(Z_OBJ_P(ZEND_THIS));
	HashTable *feeds;
	zval *requested = NULL;
	onnx_io *inputs = NULL;
	size_t input_count = 0;
	onnx_io *outputs = NULL;
	size_t output_count = 0;
	unsigned char *seen = NULL;
	OrtValue **bound = NULL;
	zval *temps = NULL;
	const char **feed_names = NULL;
	const OrtValue **feed_values = NULL;
	size_t feed_count = 0;
	char **selected = NULL;
	const char **selected_view = NULL;
	size_t selected_count = 0;
	OrtValue **results = NULL;
	zend_string *key;
	zval *feed;
	mask_job *jobs = NULL;
	int *modes = NULL;
	size_t i;

	ZEND_PARSE_PARAMETERS_START(1, 2)
		Z_PARAM_ARRAY_HT(feeds)
		Z_PARAM_OPTIONAL
		Z_PARAM_ARRAY_OR_NULL(requested)
	ZEND_PARSE_PARAMETERS_END();
	if (!require_open(session, "run")) {
		return;
	}

	if (!load_io(session->session, 0, &inputs, &input_count) || !load_io(session->session, 1, &outputs, &output_count)) {
		goto cleanup;
	}
	seen = ecalloc(input_count > 0 ? input_count : 1, 1);
	bound = ecalloc(input_count > 0 ? input_count : 1, sizeof(OrtValue *));
	temps = ecalloc(input_count > 0 ? input_count : 1, sizeof(zval));
	jobs = ecalloc(input_count > 0 ? input_count : 1, sizeof(mask_job));
	for (i = 0; i < input_count; i++) {
		ZVAL_UNDEF(&temps[i]);
	}
	ZEND_HASH_FOREACH_STR_KEY_VAL(feeds, key, feed) {
		size_t index;
		int found = 0;
		if (key == NULL) {
			onnx_throw("feed key must be an input name");
			goto cleanup;
		}
		for (index = 0; index < input_count; index++) {
			if (strcmp(inputs[index].name, ZSTR_VAL(key)) == 0) {
				found = 1;
				break;
			}
		}
		if (!found) {
			char *message;
			spprintf(&message, 0, "unknown input '%s'", ZSTR_VAL(key));
			onnx_throw_feed(message, ZSTR_VAL(key), NULL, NULL);
			efree(message);
			goto cleanup;
		}
		if (seen[index]) {
			char *message;
			spprintf(&message, 0, "input '%s' is repeated", ZSTR_VAL(key));
			onnx_throw_feed(message, ZSTR_VAL(key), NULL, NULL);
			efree(message);
			goto cleanup;
		}
		if (!classify_feed(&inputs[index], feed, &jobs[index])) {
			goto cleanup;
		}
		if (jobs[index].kind == 0 && !bind_feed(&inputs[index], feed, &bound[index], &temps[index])) {
			goto cleanup;
		}
		seen[index] = 1;
	} ZEND_HASH_FOREACH_END();
	for (i = 0; i < input_count; i++) {
		if (jobs[i].kind != 0 && !build_mask(session, &inputs[i], inputs, bound, input_count, &jobs[i], &bound[i])) {
			goto cleanup;
		}
	}
	for (i = 0; i < input_count; i++) {
		if (!seen[i] && !inputs[i].optional && !inputs[i].initializer) {
			zval expected;
			char *message;
			model_shape_zval(&inputs[i], &expected);
			spprintf(&message, 0, "input '%s' is required", inputs[i].name);
			onnx_throw_feed(message, inputs[i].name, &expected, NULL);
			efree(message);
			zval_ptr_dtor(&expected);
			goto cleanup;
		}
	}
	feed_names = ecalloc(input_count > 0 ? input_count : 1, sizeof(char *));
	feed_values = ecalloc(input_count > 0 ? input_count : 1, sizeof(OrtValue *));
	for (i = 0; i < input_count; i++) {
		if (bound[i] != NULL) {
			feed_names[feed_count] = inputs[i].name;
			feed_values[feed_count] = bound[i];
			feed_count++;
		}
	}
	if (requested == NULL || zend_hash_num_elements(Z_ARRVAL_P(requested)) == 0) {
		selected_count = output_count;
		selected = ecalloc(output_count > 0 ? output_count : 1, sizeof(char *));
		modes = ecalloc(output_count > 0 ? output_count : 1, sizeof(int));
		for (i = 0; i < output_count; i++) {
			selected[i] = dup_cstr(outputs[i].name);
		}
	} else if (!zend_array_is_list(Z_ARRVAL_P(requested))) {
		zend_string *output_key;
		zval *spec;
		selected_count = zend_hash_num_elements(Z_ARRVAL_P(requested));
		selected = ecalloc(selected_count > 0 ? selected_count : 1, sizeof(char *));
		modes = ecalloc(selected_count > 0 ? selected_count : 1, sizeof(int));
		i = 0;
		ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL_P(requested), output_key, spec) {
			size_t j;
			int found = 0;
			ZVAL_DEREF(spec);
			if (output_key == NULL) {
				onnx_throw("output key must be an output name");
				goto cleanup;
			}
			if (Z_TYPE_P(spec) != IS_ARRAY) {
				onnx_throw("output config must be an array");
				goto cleanup;
			}
			for (j = 0; j < i; j++) {
				if (strcmp(selected[j], ZSTR_VAL(output_key)) == 0) {
					char *message;
					spprintf(&message, 0, "output '%s' is repeated", ZSTR_VAL(output_key));
					onnx_throw(message);
					efree(message);
					goto cleanup;
				}
			}
			for (j = 0; j < output_count; j++) {
				if (strcmp(outputs[j].name, ZSTR_VAL(output_key)) == 0) {
					found = 1;
					break;
				}
			}
			if (!found) {
				char *message;
				spprintf(&message, 0, "unknown output '%s'", ZSTR_VAL(output_key));
				onnx_throw(message);
				efree(message);
				goto cleanup;
			}
			if (!parse_index_config(ZSTR_VAL(output_key), spec, &modes[i])) {
				goto cleanup;
			}
			selected[i++] = dup_cstr(ZSTR_VAL(output_key));
		} ZEND_HASH_FOREACH_END();
	} else {
		zval *name;
		selected_count = zend_hash_num_elements(Z_ARRVAL_P(requested));
		selected = ecalloc(selected_count > 0 ? selected_count : 1, sizeof(char *));
		modes = ecalloc(selected_count > 0 ? selected_count : 1, sizeof(int));
		i = 0;
		ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(requested), name) {
			size_t j;
			int found = 0;
			ZVAL_DEREF(name);
			if (Z_TYPE_P(name) != IS_STRING) {
				onnx_throw("output names must be strings");
				goto cleanup;
			}
			for (j = 0; j < i; j++) {
				if (strcmp(selected[j], Z_STRVAL_P(name)) == 0) {
					char *message;
					spprintf(&message, 0, "output '%s' is repeated", Z_STRVAL_P(name));
					onnx_throw(message);
					efree(message);
					goto cleanup;
				}
			}
			for (j = 0; j < output_count; j++) {
				if (strcmp(outputs[j].name, Z_STRVAL_P(name)) == 0) {
					found = 1;
					break;
				}
			}
			if (!found) {
				char *message;
				spprintf(&message, 0, "unknown output '%s'", Z_STRVAL_P(name));
				onnx_throw(message);
				efree(message);
				goto cleanup;
			}
			selected[i++] = dup_cstr(Z_STRVAL_P(name));
		} ZEND_HASH_FOREACH_END();
	}
	selected_view = ecalloc(selected_count > 0 ? selected_count : 1, sizeof(char *));
	results = ecalloc(selected_count > 0 ? selected_count : 1, sizeof(OrtValue *));
	for (i = 0; i < selected_count; i++) {
		selected_view[i] = selected[i];
	}
	if (!onnx_ort_ok(ort_api->Run(session->session, NULL, feed_names, feed_values, feed_count, selected_view, selected_count, results))) {
		goto cleanup;
	}
	array_init_size(return_value, selected_count);
	for (i = 0; i < selected_count; i++) {
		zval value;
		if (results[i] == NULL) {
			goto cleanup;
		}
		if (modes != NULL && modes[i] != 0) {
			if (!index_value(results[i], modes[i] == 2, selected[i], &value)) {
				goto cleanup;
			}
		} else if (!onnx_value_from_ort(results[i], &value)) {
			goto cleanup;
		}
		add_assoc_zval(return_value, selected[i], &value);
	}

cleanup:
	if (results != NULL) {
		for (i = 0; i < selected_count; i++) {
			if (results[i] != NULL) {
				ort_api->ReleaseValue(results[i]);
			}
		}
		efree(results);
	}
	if (selected_view != NULL) {
		efree(selected_view);
	}
	if (selected != NULL) {
		for (i = 0; i < selected_count; i++) {
			if (selected[i] != NULL) {
				efree(selected[i]);
			}
		}
		efree(selected);
	}
	if (feed_names != NULL) {
		efree(feed_names);
	}
	if (feed_values != NULL) {
		efree(feed_values);
	}
	if (bound != NULL) {
		for (i = 0; i < input_count; i++) {
			if (bound[i] != NULL) {
				ort_api->ReleaseValue(bound[i]);
			}
			if (temps != NULL && Z_TYPE(temps[i]) != IS_UNDEF) {
				zval_ptr_dtor(&temps[i]);
			}
		}
		efree(bound);
	}
	if (temps != NULL) {
		efree(temps);
	}
	if (jobs != NULL) {
		for (i = 0; i < input_count; i++) {
			if (jobs[i].lengths != NULL) {
				efree(jobs[i].lengths);
			}
			if (jobs[i].data != NULL) {
				efree(jobs[i].data);
			}
		}
		efree(jobs);
	}
	if (modes != NULL) {
		efree(modes);
	}
	if (seen != NULL) {
		efree(seen);
	}
	if (inputs != NULL) {
		for (i = 0; i < input_count; i++) {
			io_free(&inputs[i]);
		}
		efree(inputs);
	}
	if (outputs != NULL) {
		for (i = 0; i < output_count; i++) {
			io_free(&outputs[i]);
		}
		efree(outputs);
	}
}

static PHP_METHOD(OnnxValueInfo, __construct)
{
	ZEND_PARSE_PARAMETERS_NONE();
}

static PHP_METHOD(OnnxValueInfo, name)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_STR_COPY(info_from_obj(Z_OBJ_P(ZEND_THIS))->name);
}

static PHP_METHOD(OnnxValueInfo, kind)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_STR_COPY(info_from_obj(Z_OBJ_P(ZEND_THIS))->kind);
}

static PHP_METHOD(OnnxValueInfo, type)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_STR_COPY(info_from_obj(Z_OBJ_P(ZEND_THIS))->type);
}

static PHP_METHOD(OnnxValueInfo, shape)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_COPY(&info_from_obj(Z_OBJ_P(ZEND_THIS))->shape);
}

static PHP_METHOD(OnnxMetadata, __construct)
{
	ZEND_PARSE_PARAMETERS_NONE();
}

static PHP_METHOD(OnnxMetadata, producer)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_STR_COPY(metadata_from_obj(Z_OBJ_P(ZEND_THIS))->producer);
}

static PHP_METHOD(OnnxMetadata, graphName)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_STR_COPY(metadata_from_obj(Z_OBJ_P(ZEND_THIS))->graph_name);
}

static PHP_METHOD(OnnxMetadata, domain)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_STR_COPY(metadata_from_obj(Z_OBJ_P(ZEND_THIS))->domain);
}

static PHP_METHOD(OnnxMetadata, description)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_STR_COPY(metadata_from_obj(Z_OBJ_P(ZEND_THIS))->description);
}

static PHP_METHOD(OnnxMetadata, version)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_LONG(metadata_from_obj(Z_OBJ_P(ZEND_THIS))->version);
}

static PHP_METHOD(OnnxMetadata, custom)
{
	ZEND_PARSE_PARAMETERS_NONE();
	RETURN_COPY(&metadata_from_obj(Z_OBJ_P(ZEND_THIS))->custom);
}

static const zend_function_entry onnx_session_methods[] = {
	PHP_ME(OnnxSession, __construct, arginfo_session_construct, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxSession, fromContents, arginfo_session_contents, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
	PHP_ME(OnnxSession, inputs, arginfo_session_list, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxSession, outputs, arginfo_session_list, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxSession, metadata, arginfo_session_metadata, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxSession, providers, arginfo_session_list, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxSession, close, arginfo_session_void, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxSession, run, arginfo_session_run, ZEND_ACC_PUBLIC)
	PHP_FE_END
};

static const zend_function_entry onnx_value_info_methods[] = {
	PHP_ME(OnnxValueInfo, __construct, arginfo_info_construct, ZEND_ACC_PROTECTED)
	PHP_ME(OnnxValueInfo, name, arginfo_info_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValueInfo, kind, arginfo_info_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValueInfo, type, arginfo_info_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValueInfo, shape, arginfo_info_shape, ZEND_ACC_PUBLIC)
	PHP_FE_END
};

static const zend_function_entry onnx_metadata_methods[] = {
	PHP_ME(OnnxMetadata, __construct, arginfo_info_construct, ZEND_ACC_PROTECTED)
	PHP_ME(OnnxMetadata, producer, arginfo_info_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxMetadata, graphName, arginfo_info_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxMetadata, domain, arginfo_info_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxMetadata, description, arginfo_info_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxMetadata, version, arginfo_metadata_long, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxMetadata, custom, arginfo_info_shape, ZEND_ACC_PUBLIC)
	PHP_FE_END
};

static void register_handlers(zend_object_handlers *handlers, size_t offset, zend_object_free_obj_t free)
{
	memcpy(handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
	handlers->offset = offset;
	handlers->free_obj = free;
	handlers->clone_obj = NULL;
}

void onnx_register_sessions(void)
{
	zend_class_entry ce;

	INIT_CLASS_ENTRY(ce, "Switon\\OnnxRuntime\\OnnxValueInfo", onnx_value_info_methods);
	onnx_value_info_ce = zend_register_internal_class(&ce);
	onnx_value_info_ce->create_object = info_new;
	register_handlers(&onnx_value_info_handlers, XtOffsetOf(onnx_value_info_obj, std), info_free);

	INIT_CLASS_ENTRY(ce, "Switon\\OnnxRuntime\\OnnxMetadata", onnx_metadata_methods);
	onnx_metadata_ce = zend_register_internal_class(&ce);
	onnx_metadata_ce->create_object = metadata_new;
	register_handlers(&onnx_metadata_handlers, XtOffsetOf(onnx_metadata_obj, std), metadata_free);

	INIT_CLASS_ENTRY(ce, "Switon\\OnnxRuntime\\OnnxSession", onnx_session_methods);
	onnx_session_ce = zend_register_internal_class(&ce);
	onnx_session_ce->create_object = session_new;
	register_handlers(&onnx_session_handlers, XtOffsetOf(onnx_session, std), session_free);
}
