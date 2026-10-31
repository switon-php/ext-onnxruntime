<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * Identity stored in the model file.
 *
 * Label lists and task names are often entries in {@see custom()}.
 *
 * ONNX stores these fields as plain protobuf strings and an integer version.
 * An omitted string and an empty string are both `''`. An omitted version and a
 * version of zero are both `0`. This class cannot tell those apart.
 *
 * IDE stub. The extension provides this class at runtime. Do not include this file.
 */
class OnnxMetadata
{
    protected function __construct()
    {
    }

    /**
     * Tool that wrote the model file.
     *
     * An omitted field and an empty field are both `''`.
     */
    public function producer(): string
    {
    }

    /**
     * Graph name stored in the model file.
     *
     * An omitted field and an empty field are both `''`.
     */
    public function graphName(): string
    {
    }

    /**
     * ONNX domain stored in the model file.
     *
     * An omitted field and an empty field are both `''`.
     */
    public function domain(): string
    {
    }

    /**
     * Human-readable description stored in the model file.
     *
     * An omitted field and an empty field are both `''`.
     */
    public function description(): string
    {
    }

    /**
     * Model version from the ONNX graph. This is not {@see Build::version()}.
     *
     * An omitted version and a version of zero are both `0`.
     */
    public function version(): int
    {
    }

    /**
     * Custom metadata map. Keys and values are strings.
     *
     * Exporters often put label lists and task names here. A repeated key in the file
     * keeps the last value.
     *
     * @return array<string, string>
     */
    public function custom(): array
    {
    }
}
