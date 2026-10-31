<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * Base type for every failure raised by this extension.
 *
 * {@see OnnxSession} throws it when a model cannot be loaded, when an input type or shape
 * does not match the model, and when an execution provider fails a run.
 * {@see OnnxValue} throws it when a tensor cannot be built, or when {@see OnnxValue::elementType()},
 * {@see OnnxValue::shape()}, {@see OnnxValue::values()}, or {@see OnnxValue::toBytes()}
 * does not apply to that value.
 * {@see OnnxOptions} records settings. A provider name that this build cannot register
 * is reported when the session is constructed. {@see OnnxOptions::optimization()} and
 * {@see OnnxOptions::logLevel()} take enums, so a bad case cannot be passed in.
 *
 * A feed mismatch sets {@see inputName()}, {@see expectedShape()}, and {@see actualShape()}.
 * The message names that input and both shapes. A mismatch in element type alone also
 * sets all three, with both shapes equal; the message names both types. A required input
 * left out of the feed sets {@see inputName()} and {@see expectedShape()}, and leaves
 * {@see actualShape()} null. A load failure or a provider failure leaves the three
 * methods null: the message is the only context, so it names the path or the provider
 * that failed.
 *
 * IDE stub. The extension provides this class at runtime. Do not include this file.
 */
class OnnxException extends \Exception
{
    /**
     * Input name when a feed does not match the model.
     *
     * `null` for a load failure or a provider failure. The message is then the only context.
     */
    public function inputName(): ?string
    {
    }

    /**
     * Shape the model declared for {@see inputName()}.
     *
     * A dynamic dimension is its name, or `null` when the model did not name it.
     * The whole return is `null` when {@see inputName()} is `null`.
     *
     * @return list<int|string|null>|null
     */
    public function expectedShape(): ?array
    {
    }

    /**
     * Shape inferred from the feed.
     *
     * `null` when {@see inputName()} is `null`.
     *
     * @return list<int>|null
     */
    public function actualShape(): ?array
    {
    }
}
