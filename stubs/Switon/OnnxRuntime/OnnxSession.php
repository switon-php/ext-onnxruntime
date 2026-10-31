<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * A loaded model, ready to run.
 *
 * Construct it from a file path, or from the file contents with {@see fromContents()}.
 * Keep one session and call {@see run()} for each input. Creating a session compiles
 * the graph and is the slow step.
 *
 * Each feed is a {@see OnnxValue}, a PHP array, a packed string, or a small request.
 * An array or a string becomes a tensor, using that input's element type and shape.
 * The caller does not pass a shape. A PHP string is always packed bytes, never one
 * element of a string tensor. Feed strings through a list or {@see OnnxValue::tensor()}.
 * A sequence or a map is an `OnnxValue` already returned by {@see run()}.
 * An input whose name ends in `attention_mask` accepts `1` for a mask of ones.
 * Name the outputs you need in the second argument to choose which values come back.
 * A map such as `['logits' => ['index' => 'max']]` returns the winning index instead
 * of the whole tensor. One id is {@see OnnxValue::value()}.
 * The session still runs the graph. `null` and an empty list both fetch every output.
 *
 * {@see run()} blocks the PHP thread that calls it. Do not mutate a feed, or an
 * {@see OnnxValue} used as a feed, until that call has returned. Another OS thread
 * may call {@see run()} on the same session at the same time.
 *
 * IDE stub. The extension provides this class at runtime. Do not include this file.
 */
class OnnxSession
{
    /**
     * Load a model from a filesystem path.
     *
     * With no `$options`, the session uses the {@see OnnxOptions} defaults: only
     * `cpu`, thread counts left to ONNX Runtime, sequential execution,
     * graph optimization {@see OptimizationLevel::All}, and {@see LogLevel::Warning}.
     * Settings are copied from `$options` here. Later calls on that object
     * do not change this session.
     *
     * @throws OnnxException the file cannot be read, the model is not ONNX, or a provider cannot be registered
     */
    public function __construct(string $path, ?OnnxOptions $options = null)
    {
    }

    /**
     * Load a model from its file contents.
     *
     * External weights next to an `.onnx` file are not inside `$contents`. Use {@see OnnxSession::__construct()}
     * with a path when the model has external data. Defaults and the options snapshot match
     * {@see OnnxSession::__construct()}.
     *
     * This is not {@see OnnxValue::fromBytes()}. That method takes a buffer for one input
     * tensor; this one takes a whole model file.
     *
     * @throws OnnxException the contents are not an ONNX model, or a provider cannot be registered
     */
    public static function fromContents(string $contents, ?OnnxOptions $options = null): static
    {
    }

    /**
     * Model inputs, in graph order.
     *
     * Names are the keys of {@see run()}. A dynamic dimension is a string such as `batch`,
     * or `null` when the model left it unnamed. Read {@see OnnxValueInfo::shape()} before
     * packing a buffer.
     *
     * @return list<OnnxValueInfo>
     */
    public function inputs(): array
    {
    }

    /**
     * Model outputs, in graph order.
     *
     * Pass a name from here as {@see run()}'s second argument. Omitting that argument
     * returns every output listed here.
     *
     * @return list<OnnxValueInfo>
     */
    public function outputs(): array
    {
    }

    /**
     * Producer, version, description, and custom entries stored in the model file.
     *
     * This is not {@see Build::version()}, which reports the runtime. Label lists often
     * live in {@see OnnxMetadata::custom()}.
     */
    public function metadata(): OnnxMetadata
    {
    }

    /**
     * Providers registered on this session, highest precedence first.
     *
     * The default session returns `['cpu']`. Names are short: `CUDAExecutionProvider`
     * comes back as `cuda`. A name here is one this session actually loaded.
     * {@see Build::availableProviders()} can list names that fail to load.
     *
     * @return list<string>
     */
    public function providers(): array
    {
    }

    /**
     * Release the model now, instead of when the last reference goes away.
     *
     * A loaded model holds the compiled graph and its arena, which can run to hundreds
     * of megabytes. This frees them at that line, which a long-lived worker wants
     * between two models. Afterwards {@see run()}, {@see inputs()}, {@see outputs()},
     * and {@see metadata()} throw {@see OnnxException}; {@see providers()} still
     * answers. A second call does nothing. A value already returned by {@see run()}
     * keeps its own bytes.
     */
    public function close(): void
    {
    }

    /**
     * Run the model.
     *
     * `$feeds` maps each input name to a {@see OnnxValue}, a PHP array, a packed string,
     * or a small config. An array of values is copied once into native memory. A packed
     * string is read as one buffer. A config is not tensor data: the extension builds
     * the buffer.
     * The returned values keep their own bytes after this call returns. They do not hold
     * this session, so caching one does not keep the model or its arena. Dropping the
     * session does not free them, and does not leave them dangling. This method does not
     * copy their elements into PHP.
     *
     * A PHP string is packed bytes for a numeric tensor. A string tensor takes a list of
     * strings, or {@see OnnxValue::tensor()}. Feeding one PHP string to a string tensor
     * throws {@see OnnxException} with a message that says the string was read as bytes,
     * and that one element belongs in a list or in {@see OnnxValue::tensor()}.
     *
     * An input that the model marks optional, or one that already has an initializer, may
     * be left out. A required input that is missing, an unknown name, or a type or shape
     * the model rejects throws {@see OnnxException}; {@see OnnxException::inputName()} is that
     * input name.
     *
     * An input whose name ends in `attention_mask` accepts a request instead of a
     * buffer. The integer `1` fills every position with 1. The shape comes from a
     * feed already passed that shares the mask's dimension names. That buffer of
     * ones is kept on the session and reused for the same element type and shape.
     * `['sequence' => [5, 8, 3]]` marks that many leading positions in each row and
     * writes 0 after them. `['sequence' => [5, 8, 3], 'left' => true]` writes the 0s
     * on the left. Any other integer, or this config on another input, throws.
     *
     * `$outputs` chooses which outputs are returned. The graph still runs.
     * `null` and `[]` both return every output. A list of names returns those tensors.
     * A map asks for a smaller view: `['logits' => ['index' => 'max']]` returns the
     * index of the largest value on the last axis of the last step, one integer per
     * batch row. One row is one element: read it with {@see OnnxValue::value()}.
     * `['index' => 'max', 'step' => 'each']` returns that index for every step.
     * `'step'` without `'index'` throws. Ties keep
     * the first index. The result maps each requested output name to a
     * {@see OnnxValue}, in the order of `$outputs`, or in graph order when both
     * are null or empty.
     *
     * `$feeds` keys are input names. An integer key is an error, not a position. This
     * session stays usable after any {@see OnnxException} thrown here.
     *
     * @param array<string, OnnxValue|array|string|int> $feeds
     * @param list<string>|array<string, array<string, mixed>>|null $outputs
     * @return array<string, OnnxValue>
     * @throws OnnxException a feed or an output name does not match the model, or a provider fails
     */
    public function run(array $feeds, ?array $outputs = null): array
    {
    }
}
