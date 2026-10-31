<?php

declare(strict_types=1);

use Switon\OnnxRuntime\OnnxSession;
use Switon\OnnxRuntime\OnnxValue;

$session = new OnnxSession(__DIR__ . '/models/encoder.onnx');
$ids = OnnxValue::tensor([
    [101, 42, 17, 8, 102],
]);
$mask = [[1, 1, 1, 1, 1]];

$first = $session->run([
    'input_ids' => $ids,
    'attention_mask' => $mask,
], ['last_hidden_state'])['last_hidden_state'];
$second = $session->run([
    'input_ids' => $ids,
    'attention_mask' => $mask,
], ['last_hidden_state'])['last_hidden_state'];

echo json_encode($first->shape()), "\n";
echo json_encode($second->shape()), "\n";
echo $first->toBytes() === $second->toBytes() ? "same bytes\n" : "different bytes\n";
