<?php

declare(strict_types=1);

require __DIR__ . '/surface-dump.php';

if (!extension_loaded('onnxruntime')) {
    fwrite(STDERR, "onnxruntime is not loaded\n");
    exit(1);
}

$runtime = surface_namespace();
$command = escapeshellarg(PHP_BINARY) . ' -n ' . escapeshellarg(__DIR__ . '/stub-reflect.php');
$stubs = json_decode((string) shell_exec($command), true);
if (!is_array($stubs)) {
    fwrite(STDERR, "stub surface was not captured\n");
    exit(1);
}

$failures = 0;
$names = array_values(array_unique([...array_keys($runtime), ...array_keys($stubs)]));
sort($names);
foreach ($names as $class) {
    $encodedLeft = json_encode($runtime[$class] ?? null);
    $encodedRight = json_encode($stubs[$class] ?? null);
    if ($encodedLeft !== $encodedRight) {
        fwrite(STDERR, "FAIL {$class}\nruntime {$encodedLeft}\nstub    {$encodedRight}\n");
        $failures++;
    }
}
if ($failures > 0) {
    fwrite(STDERR, "{$failures} failed\n");
    exit(1);
}
echo "ok\n";
