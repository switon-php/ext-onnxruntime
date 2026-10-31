<?php

declare(strict_types=1);

use Switon\OnnxRuntime\OnnxSession;

$session = new OnnxSession(__DIR__ . '/models/resnet.onnx');
$labels = explode(',', $session->metadata()->custom()['labels'] ?? '');

$packed = file_get_contents(__DIR__ . '/models/input.f32');
if ($packed === false) {
    fwrite(STDERR, "cannot read input.f32\n");
    exit(1);
}

$logits = $session->run([
    'input' => $packed,
], ['logits'])['logits']->values();

$best = 0;
foreach ($logits as $index => $score) {
    if ($score > $logits[$best]) {
        $best = $index;
    }
}

echo ($labels[$best] ?? (string) $best), ' ', $logits[$best], "\n";
echo json_encode($logits), "\n";
