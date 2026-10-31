#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "zend_interfaces.h"
#include "ext/spl/spl_array.h"
#include "ext/json/php_json.h"
#include "php_onnxruntime.h"

#include <math.h>
#include <string.h>

static zend_object_handlers onnx_value_handlers;
static char onnx_empty_byte = 0;

static int shape_count(const int64_t *dims, size_t rank, size_t *count)
{
	size_t total = 1;
	size_t i;

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

static void value_clear_fields(onnx_value *value)
{
	value->kind = ONNX_KIND_TENSOR;
	value->from_bytes = 0;
	value->dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	value->dims = NULL;
	value->rank = 0;
	value->data = NULL;
	value->nbytes = 0;
	value->bytes = NULL;
	value->strings = NULL;
	value->nstrings = 0;
	ZVAL_UNDEF(&value->items);
	ZVAL_UNDEF(&value->entries);
	ZVAL_UNDEF(&value->map_keys);
	ZVAL_UNDEF(&value->map_values);
}

static zend_object *onnx_value_new(zend_class_entry *ce)
{
	onnx_value *value = zend_object_alloc(sizeof(onnx_value), ce);

	zend_object_std_init(&value->std, ce);
	object_properties_init(&value->std, ce);
	value->std.handlers = &onnx_value_handlers;
	value_clear_fields(value);
	return &value->std;
}

static void free_strings(zend_string **strings, size_t count)
{
	size_t i;

	if (strings == NULL) {
		return;
	}
	for (i = 0; i < count; i++) {
		if (strings[i] != NULL) {
			zend_string_release(strings[i]);
		}
	}
	efree(strings);
}

static void free_value_fields(onnx_value *value)
{
	if (value->data != NULL) {
		efree(value->data);
		value->data = NULL;
	}
	if (value->dims != NULL) {
		efree(value->dims);
		value->dims = NULL;
	}
	if (value->bytes != NULL) {
		zend_string_release(value->bytes);
		value->bytes = NULL;
	}
	free_strings(value->strings, value->nstrings);
	value->strings = NULL;
	value->nstrings = 0;
	if (Z_TYPE(value->items) != IS_UNDEF) {
		zval_ptr_dtor(&value->items);
		ZVAL_UNDEF(&value->items);
	}
	if (Z_TYPE(value->entries) != IS_UNDEF) {
		zval_ptr_dtor(&value->entries);
		ZVAL_UNDEF(&value->entries);
	}
	if (Z_TYPE(value->map_keys) != IS_UNDEF) {
		zval_ptr_dtor(&value->map_keys);
		ZVAL_UNDEF(&value->map_keys);
	}
	if (Z_TYPE(value->map_values) != IS_UNDEF) {
		zval_ptr_dtor(&value->map_values);
		ZVAL_UNDEF(&value->map_values);
	}
}

static void onnx_value_free(zend_object *object)
{
	onnx_value *value = onnx_value_from_obj(object);

	free_value_fields(value);
	zend_object_std_dtor(object);
}

onnx_value *onnx_value_create(zval *zv)
{
	object_init_ex(zv, onnx_value_ce);
	return onnx_value_from_obj(Z_OBJ_P(zv));
}

static zend_class_entry *called_value_class(void)
{
	zend_class_entry *ce = zend_get_called_scope(EG(current_execute_data));

	if (ce != NULL && instanceof_function(ce, onnx_value_ce)) {
		return ce;
	}
	return onnx_value_ce;
}

static int64_t *copy_dims(const int64_t *dims, size_t rank)
{
	int64_t *copy;

	if (rank == 0) {
		return NULL;
	}
	copy = emalloc(sizeof(int64_t) * rank);
	memcpy(copy, dims, sizeof(int64_t) * rank);
	return copy;
}

static int first_shape(zval *zv, int depth, int64_t dims[ONNX_RANK_MAX], int *rank, const char **error)
{
	zval *first;
	uint32_t count;

	ZVAL_DEREF(zv);
	if (Z_TYPE_P(zv) != IS_ARRAY) {
		*error = "tensor() expects an array";
		return 0;
	}
	if (!zend_array_is_list(Z_ARRVAL_P(zv))) {
		*error = "tensor array must be a list";
		return 0;
	}
	if (depth >= ONNX_RANK_MAX) {
		*error = "tensor rank is too high";
		return 0;
	}
	count = zend_hash_num_elements(Z_ARRVAL_P(zv));
	dims[depth] = (int64_t) count;
	*rank = depth + 1;
	if (count == 0) {
		return 1;
	}
	first = zend_hash_index_find(Z_ARRVAL_P(zv), 0);
	if (first == NULL) {
		*error = "tensor array must be a list";
		return 0;
	}
	ZVAL_DEREF(first);
	if (Z_TYPE_P(first) == IS_ARRAY) {
		return first_shape(first, depth + 1, dims, rank, error);
	}
	return 1;
}

static int check_shape(zval *zv, int depth, int rank, const int64_t *dims, const char **error)
{
	zval *element;

	ZVAL_DEREF(zv);
	if (Z_TYPE_P(zv) != IS_ARRAY || !zend_array_is_list(Z_ARRVAL_P(zv))) {
		*error = "tensor array must be a list";
		return 0;
	}
	if ((int64_t) zend_hash_num_elements(Z_ARRVAL_P(zv)) != dims[depth]) {
		*error = "rows in one dimension must be the same length";
		return 0;
	}
	if (depth + 1 == rank) {
		ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(zv), element) {
			ZVAL_DEREF(element);
			if (Z_TYPE_P(element) == IS_ARRAY) {
				*error = "rows in one dimension must be the same length";
				return 0;
			}
		} ZEND_HASH_FOREACH_END();
		return 1;
	}
	ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(zv), element) {
		ZVAL_DEREF(element);
		if (Z_TYPE_P(element) != IS_ARRAY || !check_shape(element, depth + 1, rank, dims, error)) {
			if (*error == NULL) {
				*error = "rows in one dimension must be the same length";
			}
			return 0;
		}
	} ZEND_HASH_FOREACH_END();
	return 1;
}

static int walk_kinds(zval *zv, int depth, int rank, int *saw_long, int *saw_double, int *saw_bool, int *saw_string, const char **error)
{
	zval *element;

	ZVAL_DEREF(zv);
	if (depth == rank) {
		if (Z_TYPE_P(zv) == IS_LONG) {
			*saw_long = 1;
		} else if (Z_TYPE_P(zv) == IS_DOUBLE) {
			*saw_double = 1;
		} else if (Z_TYPE_P(zv) == IS_TRUE || Z_TYPE_P(zv) == IS_FALSE) {
			*saw_bool = 1;
		} else if (Z_TYPE_P(zv) == IS_STRING) {
			*saw_string = 1;
		} else {
			*error = "tensor values must be ints, floats, bools, or strings";
			return 0;
		}
		return 1;
	}
	ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(zv), element) {
		if (!walk_kinds(element, depth + 1, rank, saw_long, saw_double, saw_bool, saw_string, error)) {
			return 0;
		}
	} ZEND_HASH_FOREACH_END();
	return 1;
}

static int long_fits_f32(zend_long value, float *out)
{
	float converted = (float) value;

	if ((zend_long) converted != value) {
		return 0;
	}
	*out = converted;
	return 1;
}

static int as_i64(zval *zv, int64_t *out, const char **error)
{
	if (Z_TYPE_P(zv) == IS_LONG) {
		*out = (int64_t) Z_LVAL_P(zv);
		return 1;
	}
	if (Z_TYPE_P(zv) == IS_DOUBLE) {
		double number = Z_DVAL_P(zv);
		int64_t converted;

		if (!isfinite(number) || number > 9223372036854774784.0 || number < -9223372036854775808.0) {
			*error = "value does not fit in an integer";
			return 0;
		}
		converted = (int64_t) number;
		if ((double) converted != number) {
			*error = "float value is not an integer";
			return 0;
		}
		*out = converted;
		return 1;
	}
	*error = "expected a number";
	return 0;
}

static int fits_range(int64_t value, int64_t low, int64_t high, const char **error)
{
	if (value < low || value > high) {
		*error = "integer is outside the element type range";
		return 0;
	}
	return 1;
}

static int write_leaf(zval *zv, ONNXTensorElementDataType type, void *data, zend_string **strings, size_t index, const char **error)
{
	int64_t integer;

	ZVAL_DEREF(zv);
	switch (type) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			if (Z_TYPE_P(zv) == IS_DOUBLE) {
				((float *) data)[index] = (float) Z_DVAL_P(zv);
				return 1;
			}
			if (Z_TYPE_P(zv) == IS_LONG) {
				float converted;
				if (!long_fits_f32(Z_LVAL_P(zv), &converted)) {
					*error = "integer does not fit in float32";
					return 0;
				}
				((float *) data)[index] = converted;
				return 1;
			}
			*error = "expected a number";
			return 0;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			if (Z_TYPE_P(zv) == IS_DOUBLE) {
				((double *) data)[index] = Z_DVAL_P(zv);
				return 1;
			}
			if (Z_TYPE_P(zv) == IS_LONG) {
				((double *) data)[index] = (double) Z_LVAL_P(zv);
				return 1;
			}
			*error = "expected a number";
			return 0;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			if (!as_i64(zv, &integer, error)) {
				return 0;
			}
			((int64_t *) data)[index] = integer;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			if (!as_i64(zv, &integer, error) || !fits_range(integer, INT32_MIN, INT32_MAX, error)) {
				return 0;
			}
			((int32_t *) data)[index] = (int32_t) integer;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			if (!as_i64(zv, &integer, error) || !fits_range(integer, INT16_MIN, INT16_MAX, error)) {
				return 0;
			}
			((int16_t *) data)[index] = (int16_t) integer;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			if (!as_i64(zv, &integer, error) || !fits_range(integer, INT8_MIN, INT8_MAX, error)) {
				return 0;
			}
			((int8_t *) data)[index] = (int8_t) integer;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			if (!as_i64(zv, &integer, error) || !fits_range(integer, 0, 255, error)) {
				return 0;
			}
			((uint8_t *) data)[index] = (uint8_t) integer;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			if (!as_i64(zv, &integer, error) || !fits_range(integer, 0, UINT16_MAX, error)) {
				return 0;
			}
			((uint16_t *) data)[index] = (uint16_t) integer;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			if (!as_i64(zv, &integer, error) || !fits_range(integer, 0, UINT32_MAX, error)) {
				return 0;
			}
			((uint32_t *) data)[index] = (uint32_t) integer;
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			if (Z_TYPE_P(zv) == IS_TRUE || (Z_TYPE_P(zv) == IS_LONG && Z_LVAL_P(zv) == 1)) {
				((uint8_t *) data)[index] = 1;
				return 1;
			}
			if (Z_TYPE_P(zv) == IS_FALSE || (Z_TYPE_P(zv) == IS_LONG && Z_LVAL_P(zv) == 0)) {
				((uint8_t *) data)[index] = 0;
				return 1;
			}
			*error = "bool element must be true, false, 0, or 1";
			return 0;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
			if (Z_TYPE_P(zv) != IS_STRING) {
				*error = "string tensor expects string elements";
				return 0;
			}
			strings[index] = zend_string_copy(Z_STR_P(zv));
			return 1;
		default:
			*error = "this element type must be packed with OnnxValue::fromBytes()";
			return 0;
	}
}

static int fill_values(zval *zv, int depth, int rank, ONNXTensorElementDataType type, void *data, zend_string **strings, size_t *index, const char **error)
{
	zval *element;

	ZVAL_DEREF(zv);
	if (depth == rank) {
		if (!write_leaf(zv, type, data, strings, *index, error)) {
			return 0;
		}
		(*index)++;
		return 1;
	}
	ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(zv), element) {
		if (!fill_values(element, depth + 1, rank, type, data, strings, index, error)) {
			return 0;
		}
	} ZEND_HASH_FOREACH_END();
	return 1;
}

static void fail_value(zval *out)
{
	if (Z_TYPE_P(out) != IS_UNDEF) {
		zval_ptr_dtor(out);
		ZVAL_UNDEF(out);
	}
}

int onnx_value_from_array(zval *values, const char *type_name, int use_model_type, ONNXTensorElementDataType model_type, zval *out, const char **error)
{
	int64_t dims[ONNX_RANK_MAX];
	int rank = 0;
	int saw_long = 0;
	int saw_double = 0;
	int saw_bool = 0;
	int saw_string = 0;
	int kinds;
	size_t count = 0;
	size_t index = 0;
	ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	onnx_value *value;
	zend_class_entry *ce;

	*error = NULL;
	ZVAL_UNDEF(out);
	if (!first_shape(values, 0, dims, &rank, error) || !check_shape(values, 0, rank, dims, error)) {
		return 0;
	}
	if (!shape_count(dims, (size_t) rank, &count)) {
		*error = "tensor shape is too large";
		return 0;
	}
	if (type_name != NULL) {
		if (!onnx_dtype_by_name(type_name, strlen(type_name), &type)) {
			*error = "unknown element type; pass a type such as float32, or pack the bytes with OnnxValue::fromBytes()";
			return 0;
		}
		if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
			*error = "float16 must be packed with OnnxValue::fromBytes()";
			return 0;
		}
	} else if (use_model_type) {
		type = model_type;
		if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
			*error = "float16 must be packed with OnnxValue::fromBytes()";
			return 0;
		}
		if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
			*error = "bfloat16 must be packed with OnnxValue::fromBytes()";
			return 0;
		}
		if (onnx_dtype_size(type) == 0 && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
			*error = "this element type must be packed with OnnxValue::fromBytes()";
			return 0;
		}
	} else if (count == 0) {
		*error = "an empty array needs a type";
		return 0;
	} else {
		if (!walk_kinds(values, 0, rank, &saw_long, &saw_double, &saw_bool, &saw_string, error)) {
			return 0;
		}
		kinds = ((saw_long || saw_double) ? 1 : 0) + saw_bool + saw_string;
		if (kinds != 1) {
			*error = "bools or strings cannot be mixed with other values";
			return 0;
		}
		if (saw_string) {
			type = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
		} else if (saw_bool) {
			type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
		} else if (saw_double) {
			type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
		} else {
			type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
		}
	}

	ce = use_model_type ? onnx_value_ce : called_value_class();
	object_init_ex(out, ce);
	value = onnx_value_from_obj(Z_OBJ_P(out));
	value->kind = ONNX_KIND_TENSOR;
	value->dtype = type;
	value->rank = (size_t) rank;
	value->dims = copy_dims(dims, (size_t) rank);
	if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		value->nstrings = count;
		value->strings = ecalloc(count > 0 ? count : 1, sizeof(zend_string *));
	} else {
		size_t element_size = onnx_dtype_size(type);
		value->nbytes = element_size * count;
		value->data = emalloc(value->nbytes > 0 ? value->nbytes : 1);
	}
	if (!fill_values(values, 0, rank, value->dtype, value->data, value->strings, &index, error)) {
		fail_value(out);
		return 0;
	}
	return 1;
}

static float f16_to_f32(uint16_t value)
{
	uint32_t sign = (uint32_t) (value & 0x8000) << 16;
	uint32_t exp = (value >> 10) & 0x1f;
	uint32_t mant = value & 0x3ff;
	uint32_t bits;
	float out;

	if (exp == 0) {
		if (mant == 0) {
			bits = sign;
		} else {
			exp = 127 - 15 + 1;
			while ((mant & 0x400) == 0) {
				mant <<= 1;
				exp--;
			}
			mant &= 0x3ff;
			bits = sign | (exp << 23) | (mant << 13);
		}
	} else if (exp == 31) {
		bits = sign | 0x7f800000u | (mant << 13);
	} else {
		bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
	}
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static float bf16_to_f32(uint16_t value)
{
	uint32_t bits = ((uint32_t) value) << 16;
	float out;

	memcpy(&out, &bits, sizeof(out));
	return out;
}

static int copy_tensor(OrtValue *src, onnx_value *value)
{
	OrtTensorTypeAndShapeInfo *info = NULL;
	size_t rank = 0;
	size_t count = 0;

	if (!onnx_ort_ok(ort_api->GetTensorTypeAndShape(src, &info))) {
		return 0;
	}
	if (!onnx_ort_ok(ort_api->GetTensorElementType(info, &value->dtype))) {
		ort_api->ReleaseTensorTypeAndShapeInfo(info);
		return 0;
	}
	if (!onnx_ort_ok(ort_api->GetDimensionsCount(info, &rank))) {
		ort_api->ReleaseTensorTypeAndShapeInfo(info);
		return 0;
	}
	value->rank = rank;
	if (rank > 0) {
		value->dims = emalloc(sizeof(int64_t) * rank);
		if (!onnx_ort_ok(ort_api->GetDimensions(info, value->dims, rank))) {
			ort_api->ReleaseTensorTypeAndShapeInfo(info);
			return 0;
		}
	}
	if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		size_t i;

		if (!onnx_ort_ok(ort_api->GetTensorShapeElementCount(info, &count))) {
			ort_api->ReleaseTensorTypeAndShapeInfo(info);
			return 0;
		}
		ort_api->ReleaseTensorTypeAndShapeInfo(info);
		value->nstrings = count;
		value->strings = ecalloc(count > 0 ? count : 1, sizeof(zend_string *));
		for (i = 0; i < count; i++) {
			size_t length = 0;
			char *buffer;

			if (!onnx_ort_ok(ort_api->GetStringTensorElementLength(src, i, &length))) {
				return 0;
			}
			buffer = emalloc(length + 1);
			if (!onnx_ort_ok(ort_api->GetStringTensorElement(src, length, i, buffer))) {
				efree(buffer);
				return 0;
			}
			buffer[length] = '\0';
			value->strings[i] = zend_string_init(buffer, length, 0);
			efree(buffer);
		}
		return 1;
	}
	ort_api->ReleaseTensorTypeAndShapeInfo(info);
	if (!onnx_ort_ok(ort_api->GetTensorSizeInBytes(src, &value->nbytes))) {
		return 0;
	}
	value->data = emalloc(value->nbytes > 0 ? value->nbytes : 1);
	if (value->nbytes > 0) {
		void *pointer = NULL;
		if (!onnx_ort_ok(ort_api->GetTensorMutableData(src, &pointer))) {
			return 0;
		}
		if (pointer != NULL) {
			memcpy(value->data, pointer, value->nbytes);
		}
	}
	return 1;
}

static int read_key(onnx_value *keys, size_t index, zval *out)
{
	if (keys->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		ZVAL_STR_COPY(out, keys->strings[index]);
		return 1;
	}
	if (keys->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
		ZVAL_LONG(out, (zend_long) ((int64_t *) keys->data)[index]);
		return 1;
	}
	if (keys->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
		ZVAL_LONG(out, ((int32_t *) keys->data)[index]);
		return 1;
	}
	onnx_throw("map key type cannot be read");
	return 0;
}

static int slice_value(onnx_value *source, size_t offset, size_t per, zval *out)
{
	onnx_value *slice = onnx_value_create(out);
	size_t element_size;

	slice->kind = ONNX_KIND_TENSOR;
	slice->dtype = source->dtype;
	slice->rank = 1;
	slice->dims = emalloc(sizeof(int64_t));
	slice->dims[0] = (int64_t) per;
	if (source->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		size_t i;

		slice->nstrings = per;
		slice->strings = ecalloc(per > 0 ? per : 1, sizeof(zend_string *));
		for (i = 0; i < per; i++) {
			slice->strings[i] = zend_string_copy(source->strings[offset + i]);
		}
		return 1;
	}
	element_size = onnx_dtype_size(source->dtype);
	if (element_size == 0) {
		onnx_throw("map value type cannot be read");
		fail_value(out);
		return 0;
	}
	slice->nbytes = element_size * per;
	slice->data = emalloc(slice->nbytes > 0 ? slice->nbytes : 1);
	if (slice->nbytes > 0) {
		memcpy(slice->data, (char *) source->data + offset * element_size, slice->nbytes);
	}
	return 1;
}

static int build_entries(onnx_value *value)
{
	onnx_value *keys = onnx_value_from_obj(Z_OBJ(value->map_keys));
	onnx_value *vals = onnx_value_from_obj(Z_OBJ(value->map_values));
	size_t key_count = 0;
	size_t value_count = 0;
	size_t per;
	size_t i;

	if (!shape_count(keys->dims, keys->rank, &key_count) || !shape_count(vals->dims, vals->rank, &value_count)) {
		onnx_throw("map shape cannot be read");
		return 0;
	}
	if (key_count == 0) {
		array_init(&value->entries);
		return 1;
	}
	if (value_count % key_count != 0) {
		onnx_throw("map keys and values do not line up");
		return 0;
	}
	per = value_count / key_count;
	array_init_size(&value->entries, key_count);
	for (i = 0; i < key_count; i++) {
		zval pair;
		zval key;
		zval item;

		array_init_size(&pair, 2);
		if (!read_key(keys, i, &key)) {
			zval_ptr_dtor(&pair);
			return 0;
		}
		if (!slice_value(vals, i * per, per, &item)) {
			zval_ptr_dtor(&key);
			zval_ptr_dtor(&pair);
			return 0;
		}
		add_next_index_zval(&pair, &key);
		add_next_index_zval(&pair, &item);
		add_next_index_zval(&value->entries, &pair);
	}
	return 1;
}

int onnx_value_from_ort(OrtValue *src, zval *out)
{
	ONNXType type;
	onnx_value *value;

	ZVAL_UNDEF(out);
	if (!onnx_ort_ok(ort_api->GetValueType(src, &type))) {
		return 0;
	}
	value = onnx_value_create(out);
	if (type == ONNX_TYPE_TENSOR) {
		value->kind = ONNX_KIND_TENSOR;
		if (!copy_tensor(src, value)) {
			fail_value(out);
			return 0;
		}
		return 1;
	}
	if (type == ONNX_TYPE_SEQUENCE) {
		size_t count = 0;
		size_t i;

		value->kind = ONNX_KIND_SEQUENCE;
		if (!onnx_ort_ok(ort_api->GetValueCount(src, &count))) {
			fail_value(out);
			return 0;
		}
		array_init_size(&value->items, count);
		for (i = 0; i < count; i++) {
			OrtValue *child = NULL;
			zval element;

			if (!onnx_ort_ok(ort_api->GetValue(src, (int) i, ort_allocator, &child))) {
				fail_value(out);
				return 0;
			}
			if (!onnx_value_from_ort(child, &element)) {
				ort_api->ReleaseValue(child);
				fail_value(out);
				return 0;
			}
			ort_api->ReleaseValue(child);
			add_next_index_zval(&value->items, &element);
		}
		return 1;
	}
	if (type == ONNX_TYPE_MAP) {
		OrtValue *keys = NULL;
		OrtValue *vals = NULL;

		value->kind = ONNX_KIND_MAP;
		if (!onnx_ort_ok(ort_api->GetValue(src, 0, ort_allocator, &keys))) {
			fail_value(out);
			return 0;
		}
		if (!onnx_ort_ok(ort_api->GetValue(src, 1, ort_allocator, &vals))) {
			ort_api->ReleaseValue(keys);
			fail_value(out);
			return 0;
		}
		if (!onnx_value_from_ort(keys, &value->map_keys) || !onnx_value_from_ort(vals, &value->map_values)) {
			ort_api->ReleaseValue(keys);
			ort_api->ReleaseValue(vals);
			fail_value(out);
			return 0;
		}
		ort_api->ReleaseValue(keys);
		ort_api->ReleaseValue(vals);
		if (!build_entries(value)) {
			fail_value(out);
			return 0;
		}
		return 1;
	}
	onnx_throw("unsupported ONNX value");
	fail_value(out);
	return 0;
}

int onnx_value_to_ort(onnx_value *value, OrtValue **out, const char **error)
{
	if (value->from_bytes) {
		*error = "packed bytes have no element type until a run binds them";
		return 0;
	}
	if (value->kind == ONNX_KIND_TENSOR) {
		if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
			size_t i;

			if (!onnx_ort_ok(ort_api->CreateTensorAsOrtValue(ort_allocator, value->dims, value->rank, value->dtype, out))) {
				return 0;
			}
			/*
			 * OrtApi::FillStringTensor would be shorter, but it takes NUL-terminated
			 * buffers and has no length argument, so an element holding a NUL byte would
			 * be cut at that byte without a word. This call takes the length instead, so
			 * a PHP string is copied byte for byte.
			 */
			for (i = 0; i < value->nstrings; i++) {
				size_t length = ZSTR_LEN(value->strings[i]);
				char *buffer = NULL;

				if (!onnx_ort_ok(ort_api->GetResizedStringTensorElementBuffer(*out, i, length, &buffer))) {
					ort_api->ReleaseValue(*out);
					*out = NULL;
					return 0;
				}
				if (length > 0 && buffer != NULL) {
					memcpy(buffer, ZSTR_VAL(value->strings[i]), length);
				}
			}
			return 1;
		}
		if (!onnx_ort_ok(ort_api->CreateTensorWithDataAsOrtValue(
				ort_cpu_mem,
				value->nbytes > 0 ? value->data : &onnx_empty_byte,
				value->nbytes,
				value->dims,
				value->rank,
				value->dtype,
				out))) {
			return 0;
		}
		return 1;
	}
	if (value->kind == ONNX_KIND_SEQUENCE) {
		size_t count;
		OrtValue **children;

		if (Z_TYPE(value->items) != IS_ARRAY) {
			*error = "sequence cannot be fed";
			return 0;
		}
		count = zend_hash_num_elements(Z_ARRVAL(value->items));
		children = ecalloc(count > 0 ? count : 1, sizeof(OrtValue *));
		size_t i = 0;
		zval *element;
		int ok = 1;

		ZEND_HASH_FOREACH_VAL(Z_ARRVAL(value->items), element) {
			ZVAL_DEREF(element);
			if (Z_TYPE_P(element) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(element), onnx_value_ce)) {
				*error = "sequence item is not an OnnxValue";
				ok = 0;
				break;
			}
			if (!onnx_value_to_ort(onnx_value_from_obj(Z_OBJ_P(element)), &children[i], error)) {
				ok = 0;
				break;
			}
			i++;
		} ZEND_HASH_FOREACH_END();
		if (ok && !onnx_ort_ok(ort_api->CreateValue((const OrtValue *const *) children, count, ONNX_TYPE_SEQUENCE, out))) {
			ok = 0;
		}
		while (i > 0) {
			i--;
			if (children[i] != NULL) {
				ort_api->ReleaseValue(children[i]);
			}
		}
		efree(children);
		return ok;
	}
	if (value->kind == ONNX_KIND_MAP) {
		OrtValue *pair[2] = {NULL, NULL};
		int ok;

		if (Z_TYPE(value->map_keys) != IS_OBJECT || Z_TYPE(value->map_values) != IS_OBJECT) {
			*error = "map cannot be fed";
			return 0;
		}
		if (!onnx_value_to_ort(onnx_value_from_obj(Z_OBJ(value->map_keys)), &pair[0], error)) {
			return 0;
		}
		if (!onnx_value_to_ort(onnx_value_from_obj(Z_OBJ(value->map_values)), &pair[1], error)) {
			ort_api->ReleaseValue(pair[0]);
			return 0;
		}
		ok = onnx_ort_ok(ort_api->CreateValue((const OrtValue *const *) pair, 2, ONNX_TYPE_MAP, out));
		ort_api->ReleaseValue(pair[0]);
		ort_api->ReleaseValue(pair[1]);
		return ok;
	}
	*error = "value cannot be fed";
	return 0;
}

static int require_tensor(onnx_value *value, const char *method)
{
	if (value->from_bytes) {
		char *message;
		spprintf(&message, 0, "OnnxValue::%s() does not apply to a fromBytes() value", method);
		onnx_throw(message);
		efree(message);
		return 0;
	}
	if (value->kind == ONNX_KIND_SEQUENCE) {
		char *message;
		spprintf(&message, 0, "OnnxValue::%s() does not apply to a sequence", method);
		onnx_throw(message);
		efree(message);
		return 0;
	}
	if (value->kind == ONNX_KIND_MAP) {
		char *message;
		spprintf(&message, 0, "OnnxValue::%s() does not apply to a map", method);
		onnx_throw(message);
		efree(message);
		return 0;
	}
	return 1;
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_value_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_debug, 0, 0, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_count, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_value_iterator, 0, 0, Traversable, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_json, 0, 0, IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_serialize, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_unserialize, 0, 1, IS_VOID, 0)
	ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_tensor, 0, 1, IS_STATIC, 0)
	ZEND_ARG_TYPE_INFO(0, values, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, type, IS_STRING, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_bytes, 0, 1, IS_STATIC, 0)
	ZEND_ARG_TYPE_INFO(0, bytes, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_string, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_value_shape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

static PHP_METHOD(OnnxValue, __construct)
{
	ZEND_PARSE_PARAMETERS_NONE();
}

static PHP_METHOD(OnnxValue, tensor)
{
	zval *values;
	zend_string *type = NULL;
	const char *error = NULL;

	ZEND_PARSE_PARAMETERS_START(1, 2)
		Z_PARAM_ARRAY(values)
		Z_PARAM_OPTIONAL
		Z_PARAM_STR_OR_NULL(type)
	ZEND_PARSE_PARAMETERS_END();
	if (!onnx_value_from_array(values, type != NULL ? ZSTR_VAL(type) : NULL, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, return_value, &error)) {
		if (error != NULL && !EG(exception)) {
			onnx_throw(error);
		}
	}
}

static PHP_METHOD(OnnxValue, fromBytes)
{
	zend_string *data;
	zend_class_entry *ce;
	onnx_value *value;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_STR(data)
	ZEND_PARSE_PARAMETERS_END();
	ce = called_value_class();
	object_init_ex(return_value, ce);
	value = onnx_value_from_obj(Z_OBJ_P(return_value));
	value->kind = ONNX_KIND_TENSOR;
	value->from_bytes = 1;
	value->bytes = zend_string_copy(data);
}

static PHP_METHOD(OnnxValue, kind)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));

	ZEND_PARSE_PARAMETERS_NONE();
	if (value->kind == ONNX_KIND_SEQUENCE) {
		RETURN_STRING("sequence");
	}
	if (value->kind == ONNX_KIND_MAP) {
		RETURN_STRING("map");
	}
	RETURN_STRING("tensor");
}

static PHP_METHOD(OnnxValue, elementType)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));

	ZEND_PARSE_PARAMETERS_NONE();
	if (!require_tensor(value, "elementType")) {
		return;
	}
	RETURN_STRING(onnx_dtype_name(value->dtype));
}

static PHP_METHOD(OnnxValue, shape)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));
	size_t i;

	ZEND_PARSE_PARAMETERS_NONE();
	if (!require_tensor(value, "shape")) {
		return;
	}
	array_init_size(return_value, value->rank);
	for (i = 0; i < value->rank; i++) {
		add_next_index_long(return_value, value->dims[i]);
	}
}

static PHP_METHOD(OnnxValue, toBytes)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));

	ZEND_PARSE_PARAMETERS_NONE();
	if (value->kind != ONNX_KIND_TENSOR) {
		onnx_throw(value->kind == ONNX_KIND_SEQUENCE
			? "OnnxValue::toBytes() does not apply to a sequence"
			: "OnnxValue::toBytes() does not apply to a map");
		return;
	}
	if (value->from_bytes) {
		RETURN_STR_COPY(value->bytes);
	}
	if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		onnx_throw("OnnxValue::toBytes() does not apply to a string tensor");
		return;
	}
	if (value->nbytes == 0) {
		RETURN_EMPTY_STRING();
	}
	RETURN_STRINGL((char *) value->data, value->nbytes);
}

static int box_element(onnx_value *value, size_t index, zval *out)
{
	switch (value->dtype) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			ZVAL_DOUBLE(out, ((float *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			ZVAL_DOUBLE(out, ((double *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
			ZVAL_DOUBLE(out, f16_to_f32(((uint16_t *) value->data)[index]));
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
			ZVAL_DOUBLE(out, bf16_to_f32(((uint16_t *) value->data)[index]));
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			ZVAL_LONG(out, (zend_long) ((int64_t *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			ZVAL_LONG(out, ((int32_t *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			ZVAL_LONG(out, ((int16_t *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			ZVAL_LONG(out, ((int8_t *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			ZVAL_LONG(out, ((uint8_t *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			ZVAL_LONG(out, ((uint16_t *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			ZVAL_LONG(out, ((uint32_t *) value->data)[index]);
			return 1;
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			ZVAL_BOOL(out, ((uint8_t *) value->data)[index] != 0);
			return 1;
		default:
			return 0;
	}
}

static PHP_METHOD(OnnxValue, values)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));
	size_t count = 0;
	size_t i;

	ZEND_PARSE_PARAMETERS_NONE();
	if (!require_tensor(value, "values")) {
		return;
	}
	if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		array_init_size(return_value, value->nstrings);
		for (i = 0; i < value->nstrings; i++) {
			add_next_index_str(return_value, zend_string_copy(value->strings[i]));
		}
		return;
	}
	if (!shape_count(value->dims, value->rank, &count)) {
		onnx_throw("OnnxValue::values() cannot read this shape");
		return;
	}
	array_init_size(return_value, count);
	for (i = 0; i < count; i++) {
		zval element;

		if (!box_element(value, i, &element)) {
			zval_ptr_dtor(return_value);
			ZVAL_UNDEF(return_value);
			onnx_throw("OnnxValue::values() cannot box this element type");
			return;
		}
		add_next_index_zval(return_value, &element);
	}
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_value_one, 0, 0, MAY_BE_LONG|MAY_BE_DOUBLE|MAY_BE_BOOL|MAY_BE_STRING)
ZEND_END_ARG_INFO()

static PHP_METHOD(OnnxValue, value)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));
	size_t count = 0;

	ZEND_PARSE_PARAMETERS_NONE();
	if (!require_tensor(value, "value")) {
		return;
	}
	if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		if (value->nstrings != 1) {
			onnx_throw("OnnxValue::value() requires exactly one element");
			return;
		}
		RETURN_STR_COPY(value->strings[0]);
	}
	if (!shape_count(value->dims, value->rank, &count) || count != 1) {
		onnx_throw("OnnxValue::value() requires exactly one element");
		return;
	}
	switch (value->dtype) {
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			RETURN_DOUBLE(((float *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			RETURN_DOUBLE(((double *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
			RETURN_DOUBLE(f16_to_f32(((uint16_t *) value->data)[0]));
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
			RETURN_DOUBLE(bf16_to_f32(((uint16_t *) value->data)[0]));
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			RETURN_LONG((zend_long) ((int64_t *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			RETURN_LONG(((int32_t *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			RETURN_LONG(((int16_t *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			RETURN_LONG(((int8_t *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			RETURN_LONG(((uint8_t *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			RETURN_LONG(((uint16_t *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			RETURN_LONG(((uint32_t *) value->data)[0]);
		case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			RETURN_BOOL(((uint8_t *) value->data)[0] != 0);
		default:
			onnx_throw("OnnxValue::value() cannot box this element type");
			return;
	}
}

static PHP_METHOD(OnnxValue, items)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));

	ZEND_PARSE_PARAMETERS_NONE();
	if (value->kind != ONNX_KIND_SEQUENCE) {
		onnx_throw("OnnxValue::items() expects a sequence");
		return;
	}
	RETURN_COPY(&value->items);
}

static PHP_METHOD(OnnxValue, entries)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));

	ZEND_PARSE_PARAMETERS_NONE();
	if (value->kind != ONNX_KIND_MAP) {
		onnx_throw("OnnxValue::entries() expects a map");
		return;
	}
	RETURN_COPY(&value->entries);
}

static const char *kind_name(onnx_value *value)
{
	if (value->kind == ONNX_KIND_SEQUENCE) {
		return "sequence";
	}
	if (value->kind == ONNX_KIND_MAP) {
		return "map";
	}
	return "tensor";
}

static void shape_to_array(onnx_value *value, zval *out)
{
	size_t i;

	array_init_size(out, value->rank);
	for (i = 0; i < value->rank; i++) {
		add_next_index_long(out, value->dims[i]);
	}
}

static void nest_tensor(onnx_value *value, size_t dim, size_t *offset, zval *out)
{
	size_t i;
	size_t length;

	if (dim == value->rank) {
		if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
			ZVAL_STR_COPY(out, value->strings[(*offset)++]);
			return;
		}
		if (!box_element(value, (*offset)++, out)) {
			ZVAL_NULL(out);
		}
		return;
	}
	length = (size_t) value->dims[dim];
	array_init_size(out, length);
	for (i = 0; i < length; i++) {
		zval child;

		nest_tensor(value, dim + 1, offset, &child);
		add_next_index_zval(out, &child);
	}
}

static void json_from_value(onnx_value *value, zval *out);

static void json_from_entry(zval *pair, zval *out)
{
	zval *key;
	zval *item;
	zval encoded;

	key = zend_hash_index_find(Z_ARRVAL_P(pair), 0);
	item = zend_hash_index_find(Z_ARRVAL_P(pair), 1);
	if (key == NULL || item == NULL) {
		return;
	}
	if (Z_TYPE_P(key) == IS_STRING) {
		json_from_value(onnx_value_from_obj(Z_OBJ_P(item)), &encoded);
		add_assoc_zval_ex(out, Z_STRVAL_P(key), Z_STRLEN_P(key), &encoded);
		return;
	}
	json_from_value(onnx_value_from_obj(Z_OBJ_P(item)), &encoded);
	add_index_zval(out, (zend_ulong) zval_get_long(key), &encoded);
}

static void json_from_value(onnx_value *value, zval *out)
{
	size_t offset = 0;
	zval *item;

	if (value->kind == ONNX_KIND_SEQUENCE) {
		array_init_size(out, zend_hash_num_elements(Z_ARRVAL(value->items)));
		ZEND_HASH_FOREACH_VAL(Z_ARRVAL(value->items), item) {
			zval encoded;

			json_from_value(onnx_value_from_obj(Z_OBJ_P(item)), &encoded);
			add_next_index_zval(out, &encoded);
		} ZEND_HASH_FOREACH_END();
		return;
	}
	if (value->kind == ONNX_KIND_MAP) {
		array_init_size(out, zend_hash_num_elements(Z_ARRVAL(value->entries)));
		ZEND_HASH_FOREACH_VAL(Z_ARRVAL(value->entries), item) {
			json_from_entry(item, out);
		} ZEND_HASH_FOREACH_END();
		return;
	}
	nest_tensor(value, 0, &offset, out);
}

static void value_take_over(onnx_value *self, onnx_value *donor)
{
	free_value_fields(self);
	self->kind = donor->kind;
	self->from_bytes = donor->from_bytes;
	self->dtype = donor->dtype;
	self->dims = donor->dims;
	self->rank = donor->rank;
	self->data = donor->data;
	self->nbytes = donor->nbytes;
	self->bytes = donor->bytes;
	self->strings = donor->strings;
	self->nstrings = donor->nstrings;
	ZVAL_COPY_VALUE(&self->items, &donor->items);
	ZVAL_COPY_VALUE(&self->entries, &donor->entries);
	ZVAL_COPY_VALUE(&self->map_keys, &donor->map_keys);
	ZVAL_COPY_VALUE(&self->map_values, &donor->map_values);
	value_clear_fields(donor);
}

static PHP_METHOD(OnnxValue, __debugInfo)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));
	size_t count = 0;

	ZEND_PARSE_PARAMETERS_NONE();
	array_init(return_value);
	add_assoc_string(return_value, "kind", (char *) kind_name(value));
	if (value->kind == ONNX_KIND_SEQUENCE) {
		add_assoc_long(return_value, "items", (zend_long) zend_hash_num_elements(Z_ARRVAL(value->items)));
		return;
	}
	if (value->kind == ONNX_KIND_MAP) {
		add_assoc_long(return_value, "entries", (zend_long) zend_hash_num_elements(Z_ARRVAL(value->entries)));
		return;
	}
	if (value->from_bytes) {
		add_assoc_bool(return_value, "fromBytes", 1);
		add_assoc_long(return_value, "bytes", (zend_long) (value->bytes != NULL ? ZSTR_LEN(value->bytes) : 0));
		return;
	}
	add_assoc_string(return_value, "type", (char *) onnx_dtype_name(value->dtype));
	if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		add_assoc_long(return_value, "elements", (zend_long) value->nstrings);
	} else if (shape_count(value->dims, value->rank, &count)) {
		add_assoc_long(return_value, "elements", (zend_long) count);
	}
	{
		zval shape;

		shape_to_array(value, &shape);
		add_assoc_zval(return_value, "shape", &shape);
	}
}

static PHP_METHOD(OnnxValue, count)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));
	size_t count = 0;

	ZEND_PARSE_PARAMETERS_NONE();
	if (value->kind == ONNX_KIND_SEQUENCE) {
		RETURN_LONG((zend_long) zend_hash_num_elements(Z_ARRVAL(value->items)));
	}
	if (value->kind == ONNX_KIND_MAP) {
		RETURN_LONG((zend_long) zend_hash_num_elements(Z_ARRVAL(value->entries)));
	}
	if (value->from_bytes) {
		onnx_throw("OnnxValue::count() does not apply to a fromBytes() value");
		return;
	}
	if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		RETURN_LONG((zend_long) value->nstrings);
	}
	if (!shape_count(value->dims, value->rank, &count)) {
		onnx_throw("OnnxValue::count() cannot read this shape");
		return;
	}
	RETURN_LONG((zend_long) count);
}

static PHP_METHOD(OnnxValue, getIterator)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));
	zval source;

	ZEND_PARSE_PARAMETERS_NONE();
	if (value->kind == ONNX_KIND_SEQUENCE) {
		ZVAL_COPY(&source, &value->items);
	} else if (value->kind == ONNX_KIND_MAP) {
		ZVAL_COPY(&source, &value->entries);
	} else {
		onnx_throw("OnnxValue::getIterator() expects a sequence or a map");
		return;
	}
	object_init_ex(return_value, spl_ce_ArrayIterator);
	zend_call_method_with_1_params(Z_OBJ_P(return_value), spl_ce_ArrayIterator, &spl_ce_ArrayIterator->constructor, "__construct", NULL, &source);
	zval_ptr_dtor(&source);
	if (EG(exception) != NULL) {
		zval_ptr_dtor(return_value);
		ZVAL_UNDEF(return_value);
	}
}

static PHP_METHOD(OnnxValue, jsonSerialize)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));

	ZEND_PARSE_PARAMETERS_NONE();
	if (value->kind == ONNX_KIND_TENSOR && value->from_bytes) {
		onnx_throw("OnnxValue::jsonSerialize() does not apply to a fromBytes() value");
		return;
	}
	if (value->kind == ONNX_KIND_TENSOR && value->dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		size_t count = 0;

		if (!shape_count(value->dims, value->rank, &count)) {
			onnx_throw("OnnxValue::jsonSerialize() cannot read this shape");
			return;
		}
		if (count == 0) {
			array_init(return_value);
			return;
		}
	}
	json_from_value(value, return_value);
}

static PHP_METHOD(OnnxValue, __serialize)
{
	onnx_value *value = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));
	zval shape;
	zval values;
	zval element;

	ZEND_PARSE_PARAMETERS_NONE();
	if (value->kind != ONNX_KIND_TENSOR) {
		onnx_throw("OnnxValue::__serialize() supports a tensor. Walk a sequence with items() or a map with entries().");
		return;
	}
	array_init(return_value);
	add_assoc_string(return_value, "kind", "tensor");
	if (value->from_bytes) {
		add_assoc_str(return_value, "packed", zend_string_copy(value->bytes));
		return;
	}
	add_assoc_string(return_value, "type", (char *) onnx_dtype_name(value->dtype));
	shape_to_array(value, &shape);
	add_assoc_zval(return_value, "shape", &shape);
	if (value->dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
		array_init_size(&values, value->nstrings);
		{
			size_t i;

			for (i = 0; i < value->nstrings; i++) {
				add_next_index_str(&values, zend_string_copy(value->strings[i]));
			}
		}
		add_assoc_zval(return_value, "values", &values);
		return;
	}
	ZVAL_STRINGL(&element, value->nbytes > 0 ? (char *) value->data : "", value->nbytes);
	add_assoc_zval(return_value, "bytes", &element);
}

static PHP_METHOD(OnnxValue, __unserialize)
{
	onnx_value *self = onnx_value_from_obj(Z_OBJ_P(ZEND_THIS));
	zval *data;
	zval *packed;
	zval *type;
	zval *shape;
	zval *bytes;
	zval *values;

	ZEND_PARSE_PARAMETERS_START(1, 1)
		Z_PARAM_ARRAY(data)
	ZEND_PARSE_PARAMETERS_END();

	packed = zend_hash_str_find(Z_ARRVAL_P(data), "packed", sizeof("packed") - 1);
	if (packed != NULL) {
		if (Z_TYPE_P(packed) != IS_STRING) {
			onnx_throw("OnnxValue::__unserialize() expects packed bytes");
			return;
		}
		{
			zval built;

			object_init_ex(&built, onnx_value_ce);
			{
				onnx_value *donor = onnx_value_from_obj(Z_OBJ_P(&built));

				donor->kind = ONNX_KIND_TENSOR;
				donor->from_bytes = 1;
				donor->bytes = zend_string_copy(Z_STR_P(packed));
			}
			value_take_over(self, onnx_value_from_obj(Z_OBJ_P(&built)));
			zval_ptr_dtor(&built);
		}
		return;
	}

	type = zend_hash_str_find(Z_ARRVAL_P(data), "type", sizeof("type") - 1);
	shape = zend_hash_str_find(Z_ARRVAL_P(data), "shape", sizeof("shape") - 1);
	values = zend_hash_str_find(Z_ARRVAL_P(data), "values", sizeof("values") - 1);
	if (type == NULL || Z_TYPE_P(type) != IS_STRING || shape == NULL || Z_TYPE_P(shape) != IS_ARRAY) {
		onnx_throw("OnnxValue::__unserialize() needs a type and a shape");
		return;
	}
	if (values != NULL) {
		zval built;
		const char *error = NULL;

		if (Z_TYPE_P(values) != IS_ARRAY) {
			onnx_throw("OnnxValue::__unserialize() expects a value list");
			return;
		}
		ZVAL_UNDEF(&built);
		if (!onnx_value_from_array(values, Z_STRVAL_P(type), 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, &built, &error)) {
			onnx_throw(error != NULL ? error : "cannot rebuild the tensor");
			return;
		}
		value_take_over(self, onnx_value_from_obj(Z_OBJ_P(&built)));
		zval_ptr_dtor(&built);
		return;
	}
	bytes = zend_hash_str_find(Z_ARRVAL_P(data), "bytes", sizeof("bytes") - 1);
	if (bytes == NULL || Z_TYPE_P(bytes) != IS_STRING) {
		onnx_throw("OnnxValue::__unserialize() needs bytes or a value list");
		return;
	}
	{
		ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		int64_t dims[ONNX_RANK_MAX];
		size_t rank = 0;
		size_t count = 0;
		size_t size;
		zval *dim;

		if (!onnx_dtype_by_name(Z_STRVAL_P(type), Z_STRLEN_P(type), &dtype)) {
			onnx_throw("OnnxValue::__unserialize() got an unknown element type");
			return;
		}
		ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape), dim) {
			if (rank >= ONNX_RANK_MAX) {
				onnx_throw("OnnxValue::__unserialize() got too many dimensions");
				return;
			}
			dims[rank++] = (int64_t) zval_get_long(dim);
		} ZEND_HASH_FOREACH_END();
		if (!shape_count(dims, rank, &count)) {
			onnx_throw("OnnxValue::__unserialize() got a bad shape");
			return;
		}
		size = onnx_dtype_size(dtype);
		if (size == 0 || count * size != Z_STRLEN_P(bytes)) {
			onnx_throw("OnnxValue::__unserialize() got a byte length that does not match the shape");
			return;
		}
		free_value_fields(self);
		self->kind = ONNX_KIND_TENSOR;
		self->from_bytes = 0;
		self->dtype = dtype;
		self->rank = rank;
		self->dims = copy_dims(dims, rank);
		self->nbytes = Z_STRLEN_P(bytes);
		self->data = emalloc(self->nbytes > 0 ? self->nbytes : 1);
		if (self->nbytes > 0) {
			memcpy(self->data, Z_STRVAL_P(bytes), self->nbytes);
		}
	}
}

static const zend_function_entry onnx_value_methods[] = {
	PHP_ME(OnnxValue, __construct, arginfo_value_construct, ZEND_ACC_PROTECTED)
	PHP_ME(OnnxValue, __debugInfo, arginfo_value_debug, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, __serialize, arginfo_value_serialize, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, __unserialize, arginfo_value_unserialize, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, tensor, arginfo_value_tensor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
	PHP_ME(OnnxValue, fromBytes, arginfo_value_bytes, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
	PHP_ME(OnnxValue, kind, arginfo_value_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, elementType, arginfo_value_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, shape, arginfo_value_shape, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, toBytes, arginfo_value_string, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, values, arginfo_value_shape, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, value, arginfo_value_one, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, items, arginfo_value_shape, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, entries, arginfo_value_shape, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, count, arginfo_value_count, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, getIterator, arginfo_value_iterator, ZEND_ACC_PUBLIC)
	PHP_ME(OnnxValue, jsonSerialize, arginfo_value_json, ZEND_ACC_PUBLIC)
	PHP_FE_END
};

void onnx_register_values(void)
{
	zend_class_entry ce;

	INIT_CLASS_ENTRY(ce, "Switon\\OnnxRuntime\\OnnxValue", onnx_value_methods);
	onnx_value_ce = zend_register_internal_class(&ce);
	onnx_value_ce->create_object = onnx_value_new;
	memcpy(&onnx_value_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
	onnx_value_handlers.offset = XtOffsetOf(onnx_value, std);
	onnx_value_handlers.free_obj = onnx_value_free;
	onnx_value_handlers.clone_obj = NULL;
	zend_class_implements(onnx_value_ce, 3, zend_ce_countable, zend_ce_aggregate, php_json_serializable_ce);
}
