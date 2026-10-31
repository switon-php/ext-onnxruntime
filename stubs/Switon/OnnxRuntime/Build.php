<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * The ONNX Runtime build linked into this extension.
 *
 * Call {@see version()} and {@see availableProviders()} before opening a {@see OnnxSession}
 * when the process must choose CUDA or CoreML only if that build actually has it.
 *
 * The class is `Build` because the namespace already says `OnnxRuntime`. Its methods
 * describe the binary, not a session: a session that failed to register `cuda` still
 * lists it here when the build was compiled with it.
 *
 * IDE stub. The extension provides this class at runtime. Do not include this file.
 */
class Build
{
    protected function __construct()
    {
    }

    /**
     * ONNX Runtime version linked into this extension, such as `1.31.1`.
     */
    public static function version(): string
    {
    }

    /**
     * Execution providers this build was compiled with, in the order it reports them.
     * That order is not a priority. {@see OnnxOptions::providers()} sets priority.
     *
     * Names come back short. Each one can be passed to {@see OnnxOptions::providers()}.
     * The id ending in `ExecutionProvider` is the same name, such as
     * `CUDAExecutionProvider` for `cuda`. An entry is not a promise that it can load.
     * Pass a subset only when this machine has that provider's libraries.
     * The default session uses CPU alone.
     *
     * @return list<string>
     */
    public static function availableProviders(): array
    {
    }
}
