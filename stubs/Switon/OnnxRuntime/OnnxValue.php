<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * One model input or output kept inside the extension.
 *
 * A tensor's bytes stay off the PHP heap. {@see tensor()} copies a PHP array once, one
 * element at a time, straight into native memory. The element
 * type follows the values unless the caller passes one: ints, floats, bools, or strings.
 * The shape follows the nesting. A list is one dimension. A list of lists is two.
 * When {@see OnnxSession::run()} binds it to an input whose rank is one higher, the
 * missing dimension is a batch of 1, provided that dimension is dynamic. A fixed
 * dimension that does not match is an error.
 *
 * A value from {@see tensor()} carries an element type. A PHP array, a PHP string, and a
 * value from {@see fromBytes()} carry none. An untyped feed takes the element type the
 * model declares, and its values are converted to that type by the same rules
 * {@see tensor()} applies: nothing is truncated or clamped, and a lossy value throws. A
 * typed feed must already match. {@see OnnxSession::run()} throws {@see OnnxException}
 * on a mismatch.
 *
 * {@see fromBytes()} takes a buffer packed for one tensor input. The element type decides
 * how many bytes each element takes. The shape decides how many elements there are.
 * This object does not record them. {@see elementType()}, {@see shape()}, and {@see values()}
 * throw on it. {@see OnnxSession::run()} checks the length against the input and does not
 * write that answer back. Pass the same instance into later runs. Each run reads the buffer
 * and does not copy it again.
 *
 * A value returned by {@see OnnxSession::run()} keeps its own bytes. Dropping the session
 * does not free it, and keeping it does not keep the model loaded.
 *
 * A sequence or a map is also an `OnnxValue`, returned by {@see OnnxSession::run()}.
 * There is no factory for one. Pass that value into a later {@see OnnxSession::run()}
 * when the input expects the same container. {@see items()} and {@see entries()} walk
 * them. {@see values()} and {@see toBytes()} are for a tensor. {@see OnnxValueInfo::type()}
 * prints a declaration such as `sequence<float32>` in one string. Here the container is
 * {@see kind()} and the element type is {@see elementType()}.
 *
 * The nesting is one level. ONNX Runtime registers no type for a sequence of sequences,
 * so a model that declares `sequence<sequence<float32>>` fails to load before this class
 * is involved, in any language. A sequence of maps is not a case this extension has to
 * answer for, because the runtime does not offer it either.
 *
 * The class speaks the PHP data protocols, so a value is visible to code that does not
 * know this extension. {@see __debugInfo()} shows kind, type, and shape under `var_dump()`.
 * {@see jsonSerialize()} rebuilds the nested arrays, so `json_encode()` returns the tensor
 * as it looks in the model. {@see count()} is the element count of a tensor, or the number
 * of items or entries. A sequence and a map are iterable; a tensor is not, and `foreach`
 * on one throws instead of looping zero times. {@see __serialize()} stores the element
 * type, the shape, and the bytes, so a value survives a cache round trip. A sequence and
 * a map are not serialized: walk them first.
 *
 * IDE stub. The extension provides this class at runtime. Do not include this file.
 */
class OnnxValue implements \Countable, \IteratorAggregate, \JsonSerializable
{
    /**
     * 32-bit float. {@see tensor()} infers this from a list of floats.
     */
    public const FLOAT32 = 'float32';

    /**
     * 16-bit float, two bytes per element. Compare with {@see elementType()}, or pack with
     * {@see fromBytes()}. {@see tensor()} does not accept it.
     */
    public const FLOAT16 = 'float16';

    /**
     * 64-bit integer. {@see tensor()} infers this from a list of ints.
     */
    public const INT64 = 'int64';

    /**
     * 32-bit integer. Pass it as {@see tensor()}'s `$type`. Ints alone infer {@see INT64}.
     */
    public const INT32 = 'int32';

    /**
     * 8-bit signed integer. Pass it as {@see tensor()}'s `$type`.
     */
    public const INT8 = 'int8';

    /**
     * 8-bit unsigned integer. Pass it as {@see tensor()}'s `$type`.
     */
    public const UINT8 = 'uint8';

    /**
     * Boolean. {@see tensor()} infers this from a list of bools.
     */
    public const BOOL = 'bool';

    /**
     * String element. {@see tensor()} infers this from a list of strings.
     * {@see toBytes()} does not apply. Use {@see values()}.
     */
    public const STRING = 'string';

    /**
     * A tensor. Returned by {@see kind()}.
     */
    public const TENSOR = 'tensor';

    /**
     * A sequence of values. Returned by {@see kind()}. Walk it with {@see items()}.
     */
    public const SEQUENCE = 'sequence';

    /**
     * A map of values. Returned by {@see kind()}. Walk it with {@see entries()}.
     */
    public const MAP = 'map';

    protected function __construct()
    {
    }

    /**
     * Tensor from a PHP array.
     *
     * With `$type` omitted, a list of ints is int64 and a list of floats is float32.
     * Ints mixed with floats become float32. Bools and strings follow the same way.
     * An empty array without `$type`, or a mix of bools or strings with other values,
     * throws {@see OnnxException}. Pass `$type` to force a constant such as {@see INT32}
     * or {@see UINT8}; every value is then converted to it. A float with a fractional
     * part into an integer type, or an integer outside that type's range, throws
     * {@see OnnxException}. Values are not truncated or clamped.
     *
     * `float64`, `int16`, `uint16`, and `uint32` are accepted by name; they have no
     * constant. {@see FLOAT16}, `bfloat16`, and `uint64` are not accepted here: PHP has
     * no 16-bit float, and a PHP int cannot hold the top half of a uint64. Pack those
     * with {@see fromBytes()}.
     *
     * A string element may hold any bytes, including a NUL and anything above `0x7f`.
     * The bytes are copied as they are: not decoded, and not checked for UTF-8. Each
     * element carries its own length, so nothing is cut short at a NUL.
     *
     * Nesting is the shape: `$ids` is one dimension, `[$ids, $other]` is two.
     * Rows in one dimension must be the same length. An empty list with `$type` set
     * has shape `[0]`. {@see OnnxSession::run()} throws {@see OnnxException} when the
     * rank still does not match the model input.
     *
     * A rank-0 input cannot be built here: a PHP list is one dimension, so `[5.0]` is
     * shape `[1]`. Feed a scalar input with {@see fromBytes()} and one element's worth
     * of bytes, such as `pack('f', 5.0)` for float32. A bare PHP number is not a feed.
     *
     * @param array<int|float|bool|string|array> $values
     * @param self::FLOAT32|self::INT64|self::INT32|self::INT8|self::UINT8|self::BOOL|self::STRING|'float64'|'int16'|'uint16'|'uint32'|null $type
     */
    public static function tensor(array $values, ?string $type = null): static
    {
    }

    /**
     * Tensor from a packed buffer.
     *
     * `$bytes` is row-major, in native byte order, with no header and no padding. The
     * element type and the element count come from the model input this value is fed to,
     * not from this object. {@see elementType()} and {@see shape()} throw {@see OnnxException}
     * on this instance, including after {@see OnnxSession::run()}. The byte length must
     * equal the input's element size times its element count. One dynamic dimension may
     * be filled from that length. Two or more leave the shape undetermined, and the run
     * throws. Float16 stays two bytes per element and is not converted. {@see toBytes()}
     * returns this same buffer.
     *
     * A PHP string passed straight to {@see OnnxSession::run()} is this buffer, not a
     * string-tensor element. Build string tensors with {@see tensor()}.
     */
    public static function fromBytes(string $bytes): static
    {
    }

    /**
     * `tensor`, `sequence`, or `map`.
     *
     * Compare with {@see TENSOR}, {@see SEQUENCE}, and {@see MAP}. This is the container.
     * The element type is {@see elementType()}. A {@see fromBytes()} value is a tensor, so this
     * returns {@see TENSOR} before any run binds it.
     */
    public function kind(): string
    {
    }

    /**
     * Element type for a tensor, such as `float32` or `int64`.
     *
     * The constants on this class are the types {@see tensor()} accepts. A model may
     * still report another string, such as `float64` or `bfloat16`.
     *
     * Throws {@see OnnxException} when this value is a sequence or a map, or when it
     * was created with {@see fromBytes()}.
     */
    public function elementType(): string
    {
    }

    /**
     * Concrete shape of a tensor.
     *
     * Throws {@see OnnxException} when this value is a sequence or a map, or when it
     * was created with {@see fromBytes()}.
     *
     * @return list<int>
     */
    public function shape(): array
    {
    }

    /**
     * Raw element bytes of a tensor.
     *
     * One buffer, not one PHP value per element. Row-major, in native byte order, with
     * no header and no padding. On a value from {@see OnnxSession::run()} the length is
     * the product of {@see shape()} times the element size, and these bytes belong to the
     * value, not to the session. On a {@see fromBytes()} value this is the buffer that was
     * passed in. Float16 stays two bytes per element.
     * Throws {@see OnnxException} for a string tensor, a sequence, or a map. Use
     * {@see values()} for a string tensor.
     */
    public function toBytes(): string
    {
    }

    /**
     * Copy a tensor into a PHP list, row-major and flat.
     *
     * Shape is not rebuilt as nested arrays. A `[batch, sequence, hidden]` tensor
     * comes back as one list of `batch * sequence * hidden` values. Pair it with
     * {@see shape()} to cut the list. This allocates one PHP value per element.
     * Use it for a short vector, such as class scores. Read a hidden state or an
     * image-sized tensor with {@see toBytes()}.
     *
     * Float16 and bfloat16 are widened to a PHP float. Throws {@see OnnxException} for a
     * sequence, a map, a {@see fromBytes()} value, or an element type PHP has no value
     * for, such as float8 or int4.
     *
     * @return list<float|int|bool|string>
     */
    public function values(): array
    {
    }

    /**
     * The one element in this tensor.
     *
     * Returns an int, float, bool, or string. Throws {@see OnnxException} when the
     * tensor has any other number of elements, or when {@see values()} cannot box
     * the element type. A sequence, a map, and a {@see fromBytes()} value throw as well.
     */
    public function value(): int|float|bool|string
    {
    }

    /**
     * OnnxValues inside a sequence, in order. This does not copy tensor elements into PHP.
     *
     * Throws {@see OnnxException} when this value is not a sequence.
     *
     * @return list<OnnxValue>
     */
    public function items(): array
    {
    }

    /**
     * Entries inside a map.
     *
     * Each entry is a key and an `OnnxValue`. Keys are strings or ints, matching the model.
     * This does not copy tensor elements into PHP. A scalar score is a tensor of one
     * element: read it with {@see value()}.
     * Throws {@see OnnxException} when this value is not a map.
     *
     * @return list<array{0: int|string, 1: OnnxValue}>
     */
    public function entries(): array
    {
    }

    /**
     * What `var_dump()` prints for this value.
     *
     * A tensor reports `kind`, `type`, `elements`, and `shape`. A {@see fromBytes()} value
     * reports `kind`, `fromBytes`, and the buffer length, because it carries no type or
     * shape. A sequence reports `kind` and `items`; a map reports `kind` and `entries`.
     * This method reads no tensor element, so it stays cheap on a large tensor.
     *
     * @return array<string, int|string|bool|list<int>>|null
     */
    public function __debugInfo(): ?array
    {
    }

    /**
     * State kept by `serialize()`.
     *
     * A tensor stores its element type, its shape, and its bytes, or its strings for a
     * string tensor. A {@see fromBytes()} value stores the buffer alone. {@see __unserialize()}
     * rebuilds the same value, so a tensor can go into a cache and come back.
     *
     * Throws {@see OnnxException} for a sequence or a map: walk them with {@see items()}
     * or {@see entries()} first.
     *
     * @return array<string, mixed>
     */
    public function __serialize(): array
    {
    }

    /**
     * Rebuild a value from the state {@see __serialize()} wrote.
     *
     * `$data` is that array. A value rebuilt from bytes reports its type and shape, so
     * {@see elementType()}, {@see shape()}, {@see values()}, and {@see toBytes()} all work.
     *
     * @param array<string, mixed> $data
     * @throws OnnxException the array is not the shape this class writes
     */
    public function __unserialize(array $data): void
    {
    }

    /**
     * Number of elements, items, or entries.
     *
     * A tensor reports the product of {@see shape()}, and a string tensor the number of
     * strings. A sequence reports its item count, a map its entry count. Throws
     * {@see OnnxException} for a {@see fromBytes()} value, whose shape is unknown.
     */
    public function count(): int
    {
    }

    /**
     * Iterator over a sequence or a map, for `foreach`.
     *
     * A sequence yields its {@see items()}, in order. A map yields its {@see entries()},
     * each a `[key, value]` pair. Throws {@see OnnxException} for a tensor: use
     * {@see values()} and {@see shape()}, or {@see jsonSerialize()}.
     *
     * @return \Traversable<int, OnnxValue|array{0: int|string, 1: OnnxValue}>
     */
    public function getIterator(): \Traversable
    {
    }

    /**
     * Nested PHP arrays for `json_encode()`.
     *
     * A tensor becomes the nesting its {@see shape()} describes, so a `[2, 2]` tensor is
     * two lists of two values. A sequence becomes a list, and a map becomes an object keyed
     * by its entries. This allocates one PHP value per element, like {@see values()}.
     *
     * Throws {@see OnnxException} for a {@see fromBytes()} value, whose shape is unknown.
     *
     * @return array<int|string, mixed>|int|float|bool|string|null
     */
    public function jsonSerialize(): mixed
    {
    }
}
