<?php

declare(strict_types=1);

use Switon\OnnxRuntime\Build;
use Switon\OnnxRuntime\OnnxSession;

echo Build::version(), "\n";

foreach (Build::availableProviders() as $provider) {
    echo $provider, "\n";
}

$session = new OnnxSession(__DIR__ . '/models/encoder.onnx');

foreach ($session->inputs() as $input) {
    echo $input->kind(), ' ', $input->name(), ' ', $input->type(), ' ', json_encode($input->shape()), "\n";
}

foreach ($session->outputs() as $output) {
    echo $output->name(), ' ', $output->type(), "\n";
}

echo implode(', ', $session->providers()), "\n";

$labels = $session->metadata()->custom()['labels'] ?? '';
echo $labels, "\n";
