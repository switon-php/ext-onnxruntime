<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * Least severe message a session log keeps.
 *
 * {@see Warning} is the default.
 *
 * IDE stub. The extension provides this enum at runtime. Do not include this file.
 */
enum LogLevel: string
{
    /**
     * Most detailed session log. Pair with {@see OnnxOptions::verbosity()}.
     */
    case Verbose = 'verbose';

    /**
     * Informational messages and worse.
     */
    case Info = 'info';

    /**
     * Warnings and worse. This is the default.
     */
    case Warning = 'warning';

    /**
     * Errors and fatal messages.
     */
    case Error = 'error';

    /**
     * Fatal messages only.
     */
    case Fatal = 'fatal';
}
