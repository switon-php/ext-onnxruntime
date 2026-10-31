<?php

declare(strict_types=1);

require __DIR__ . '/surface-dump.php';

foreach (glob(__DIR__ . '/../stubs/Switon/OnnxRuntime/*.php') as $file) {
    require $file;
}

echo json_encode(surface_namespace(), JSON_THROW_ON_ERROR), "\n";
