<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * Name, kind, type, and shape of one model input or output.
 *
 * This describes an input before it is fed. It holds no data. Read it to learn what
 * shape to build. {@see OnnxValue} is the data itself.
 *
 * A free dimension is a name such as `batch` in {@see shape()}, or `null` when the model
 * does not name it. A concrete {@see OnnxValue} passed to {@see OnnxSession::run()} still
 * has to pick a size for that dimension.
 *
 * IDE stub. The extension provides this class at runtime. Do not include this file.
 */
class OnnxValueInfo
{
    protected function __construct()
    {
    }

    /**
     * Input or output name.
     *
     * Use an input name as a key in {@see OnnxSession::run()}. Use an output name in
     * that method's second argument.
     */
    public function name(): string
    {
    }

    /**
     * `tensor`, `sequence`, or `map`.
     *
     * Same strings as {@see OnnxValue::TENSOR}, {@see OnnxValue::SEQUENCE}, and
     * {@see OnnxValue::MAP}. The element type is {@see type()}, not this method.
     */
    public function kind(): string
    {
    }

    /**
     * Declaration of this input or output, in one string.
     *
     * A tensor is an element type such as `float32` or `int64`. A sequence or a map
     * includes the container, for example `sequence<float32>` or `map<int64,float32>`.
     * The string can name a type that is not an {@see OnnxValue} constant, such as
     * `float64`. {@see kind()} is the container alone. A live {@see OnnxValue} splits
     * the same fact into {@see OnnxValue::kind()} and {@see OnnxValue::elementType()}.
     *
     * A PHP array or string passed to {@see OnnxSession::run()} becomes a tensor.
     * Feed a sequence or a map by passing the {@see OnnxValue} {@see OnnxSession::run()}
     * returned.
     */
    public function type(): string
    {
    }

    /**
     * Declared shape. A fixed size is an int. A named dynamic dimension is its name,
     * such as `batch`. An unnamed dynamic dimension is `null`.
     *
     * Empty for a sequence or a map; use {@see type()} for those.
     *
     * A dimension that {@see OnnxOptions::freeDimensions()} fixes is reported as that
     * number, not as the name.
     *
     * @return list<int|string|null>
     */
    public function shape(): array
    {
    }
}
