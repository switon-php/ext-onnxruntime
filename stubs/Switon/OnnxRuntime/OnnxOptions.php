<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * How a {@see OnnxSession} loads and runs a model.
 *
 * Omit this and the session uses only `cpu`. A name from
 * {@see Build::availableProviders()} is compiled in, not proven to load:
 * `cuda` fails when cuDNN is missing. Pass that name to {@see providers()} only
 * on a machine that can run it. Thread counts stay with ONNX Runtime, execution
 * is sequential, graph optimization is {@see OptimizationLevel::All}, and logs sit at
 * {@see LogLevel::Warning}.
 *
 * Each call replaces the previous value on this object. {@see OnnxSession} copies
 * the settings when it is constructed. Changing this object later does not
 * change a session already created from it. One options object can still be
 * passed to several sessions; each copies the settings it sees at construction.
 *
 * IDE stub. The extension provides this class at runtime. Do not include this file.
 */
class OnnxOptions
{
    /**
     * Start from the defaults: `cpu`, runtime-chosen thread counts, sequential
     * execution, {@see OptimizationLevel::All}, and {@see LogLevel::Warning}.
     */
    public function __construct()
    {
    }

    /**
     * Threads used inside one graph node. `0` lets ONNX Runtime choose. A negative
     * count is rejected when the session is constructed.
     *
     * This is ONNX Runtime's `intra_op_num_threads`. The session copies this value
     * when it is constructed.
     */
    public function intraOpThreads(int $threads): static
    {
    }

    /**
     * Threads used across graph nodes. `0` lets ONNX Runtime choose. A negative count
     * is rejected when the session is constructed.
     *
     * This is ONNX Runtime's `inter_op_num_threads`. Used when {@see parallel()} is
     * enabled.
     */
    public function interOpThreads(int $threads): static
    {
    }

    /**
     * How far to optimize the graph while the session loads.
     */
    public function optimization(OptimizationLevel $level): static
    {
    }

    /**
     * Overlap independent graph nodes.
     *
     * The default runs one node at a time. {@see interOpThreads()} applies
     * once this is enabled.
     */
    public function parallel(bool $parallel = true): static
    {
    }

    /**
     * Least severe message the session log keeps.
     */
    public function logLevel(LogLevel $level): static
    {
    }

    /**
     * Extra detail once {@see logLevel()} is {@see LogLevel::Verbose}. `0` is the default.
     * A negative level is rejected when the session is constructed.
     */
    public function verbosity(int $level): static
    {
    }

    /**
     * Execution providers to try, first match wins for each node.
     *
     * Each entry is a name, or a pair of that name and string options.
     * A short name from {@see Build::availableProviders()} can be passed back here.
     * So can the id that ends in `ExecutionProvider`. `cuda` and
     * `CUDAExecutionProvider` are the same name. So are `cpu` and
     * `CPUExecutionProvider`, `coreml` and `CoreMLExecutionProvider`, and
     * `tensorrt` and `TensorrtExecutionProvider`. Those four are accepted even
     * when this build does not list them; the runtime then reports the failure.
     * `['cuda', 'cpu']` runs a node on CUDA when it can, and on CPU otherwise.
     * `[['cuda', ['device_id' => '0']], 'cpu']` passes that map through.
     * Keys are not interpreted here. Every value is a string, and the provider
     * receives the map as its own options. `device_id` is one CUDA key.
     * {@see Build::availableProviders()} and {@see OnnxSession::providers()}
     * return the short form. No `cpu` entry is added for you: a list without `cpu` has
     * no CPU fallback. An empty list, or a name this build cannot register, is rejected
     * when the session is constructed.
     *
     * @param list<string|array{0: string, 1: array<string, string>}> $providers
     */
    public function providers(array $providers): static
    {
    }

    /**
     * Fix a free dimension by name.
     *
     * A model that leaves two or more dimensions free cannot take a packed string:
     * the byte length alone does not say how it splits. Naming a size here takes that
     * dimension out of the guess, so the buffer only has to resolve the rest.
     * `['batch' => 1]` fixes `batch`, and a buffer then fills the other free dimension
     * from its length. The name is the one {@see OnnxValueInfo::shape()} reports, such as
     * `batch`. A name the model does not declare is ignored.
     *
     * A size must be a positive integer. The session copies the list when it is
     * constructed, like every other setting. {@see OnnxValueInfo::shape()} then reports that
     * number instead of the name.
     *
     * @param array<string, int> $dimensions
     */
    public function freeDimensions(array $dimensions): static
    {
    }
}
