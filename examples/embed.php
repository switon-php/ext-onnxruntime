<?php

declare(strict_types=1);

use Switon\OnnxRuntime\OnnxOptions;
use Switon\OnnxRuntime\Build;
use Switon\OnnxRuntime\OnnxSession;

$providers = array_values(array_filter(
    ['cuda', 'cpu'],
    static fn (string $name): bool => in_array($name, Build::availableProviders(), true),
));
$options = (new OnnxOptions())
    ->providers($providers)
    ->intraOpThreads(4);

$session = new OnnxSession(__DIR__ . '/models/encoder.onnx', $options);

$ids = [
    [101, 42, 17, 8, 102],
    [101, 7, 19, 0, 0],
];
$mask = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0],
];

$hidden = $session->run([
    'input_ids' => $ids,
    'attention_mask' => $mask,
], ['last_hidden_state'])['last_hidden_state'];

[$batch, $seq, $width] = $hidden->shape();
$flat = $hidden->values();
echo json_encode($hidden->shape()), ' ', strlen($hidden->toBytes()), " bytes\n";

for ($row = 0; $row < $batch; $row++) {
    $sum = array_fill(0, $width, 0.0);
    $count = 0;
    for ($token = 0; $token < $seq; $token++) {
        if ($mask[$row][$token] === 0) {
            continue;
        }
        $count++;
        $offset = ($row * $seq + $token) * $width;
        for ($dim = 0; $dim < $width; $dim++) {
            $sum[$dim] += $flat[$offset + $dim];
        }
    }
    $pooled = array_map(static fn (float $value): float => $value / $count, $sum);
    echo json_encode($pooled), "\n";
}
