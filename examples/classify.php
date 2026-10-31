<?php

declare(strict_types=1);

use Switon\OnnxRuntime\OnnxSession;

$session = new OnnxSession(__DIR__ . '/models/classifier.onnx');
$labels = explode(',', $session->metadata()->custom()['labels'] ?? '');

$row = [0.2, 1.5, -0.3, 0.4];
$logits = $session->run(['input' => $row], ['logits'])['logits']->values();

// The loop keeps the score for the line below. When only the label is needed,
// ask for the index instead of copying every score into PHP:
// $best = $session->run(['input' => $row], ['logits' => ['index' => 'max']])['logits']->value();
$best = 0;
foreach ($logits as $index => $score) {
    if ($score > $logits[$best]) {
        $best = $index;
    }
}

echo ($labels[$best] ?? (string) $best), ' ', $logits[$best], "\n";
echo json_encode($logits), "\n";
