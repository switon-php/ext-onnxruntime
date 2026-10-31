<?php

declare(strict_types=1);

namespace Switon\OnnxRuntime;

/**
 * How far to optimize a graph while a session loads.
 *
 * {@see All} includes layout changes. There is no separate layout level.
 *
 * IDE stub. The extension provides this enum at runtime. Do not include this file.
 */
enum OptimizationLevel: string
{
    /**
     * Leave the graph as it was loaded.
     */
    case Off = 'off';

    /**
     * Basic rewrites only.
     */
    case Basic = 'basic';

    /**
     * Basic rewrites plus the extended pass.
     */
    case Extended = 'extended';

    /**
     * Every optimization, including layout changes. This is the default.
     */
    case All = 'all';
}
