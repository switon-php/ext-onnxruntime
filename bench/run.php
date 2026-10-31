<?php

declare(strict_types=1);

/**
 * What a PHP process pays to run one inference, in-process versus over HTTP.
 *
 * The comparison everyone actually faces: keep the model in a Python service and
 * post a tensor to it, or load the model in the PHP process. Both sides run the same
 * graph on the same ONNX Runtime, pinned to the same thread counts, on the same
 * machine, in the same second. What differs is the boundary.
 *
 *   php -d extension=$PWD/pie/modules/onnxruntime.so bench/run.php
 *
 * The Python side needs bench/.venv (see bench/README.md). This script starts it,
 * waits for /health, measures, and stops it.
 */

use Switon\OnnxRuntime\Build;
use Switon\OnnxRuntime\OnnxOptions;
use Switon\OnnxRuntime\OnnxSession;
use Switon\OnnxRuntime\OnnxValue;
use Switon\OnnxRuntime\OptimizationLevel;

const THREADS = 4;
const WARMUP = 30;
const ITERATIONS = 200;
const PORT = 8731;

$root = dirname(__DIR__);
$models = __DIR__ . '/models';

/** Nearest-rank percentile over an ascending sample list. */
function percentile(array $sorted, float $p): float
{
    $rank = (int) ceil($p / 100 * count($sorted)) - 1;
    return $sorted[max(0, min($rank, count($sorted) - 1))];
}

/**
 * Time one callable. Every sample is kept, so the tail is real and not an average of
 * averages.
 *
 * @return array{p50: float, p90: float, p99: float, mean: float}
 */
function measure(callable $call, int $warmup = WARMUP, int $iterations = ITERATIONS): array
{
    for ($i = 0; $i < $warmup; $i++) {
        $call();
    }
    $samples = [];
    for ($i = 0; $i < $iterations; $i++) {
        $start = hrtime(true);
        $call();
        $samples[] = hrtime(true) - $start;
    }
    sort($samples);
    $ms = static fn (float $ns): float => $ns / 1e6;
    return [
        'p50' => $ms(percentile($samples, 50)),
        'p90' => $ms(percentile($samples, 90)),
        'p99' => $ms(percentile($samples, 99)),
        'mean' => $ms(array_sum($samples) / count($samples)),
    ];
}

function row(string $label, array $m): void
{
    printf("  %-46s %8.3f %8.3f %8.3f %8.3f\n", $label, $m['p50'], $m['p90'], $m['p99'], $m['mean']);
}

function section(string $title): void
{
    echo "\n", $title, "\n";
    printf("  %-46s %8s %8s %8s %8s\n", '', 'p50', 'p90', 'p99', 'mean');
}

// ---------------------------------------------------------------- python server

$python = getenv('BENCH_PYTHON') ?: __DIR__ . '/.venv/bin/python';
if (!is_file($python)) {
    fwrite(STDERR, "no Python at {$python}. See bench/README.md.\n");
    exit(1);
}

$server = proc_open(
    [$python, __DIR__ . '/server.py', '--port', (string) PORT, '--threads', (string) THREADS],
    [1 => ['file', '/dev/null', 'w'], 2 => ['file', '/dev/null', 'w']],
    $pipes,
);

$base = 'http://127.0.0.1:' . PORT;
$ready = false;
for ($i = 0; $i < 300; $i++) {
    $probe = @file_get_contents($base . '/health');
    if ($probe !== false) {
        $ready = true;
        break;
    }
    usleep(100_000);
}
if (!$ready) {
    fwrite(STDERR, "the Python server did not come up\n");
    proc_terminate($server);
    exit(1);
}

$http = curl_init();
curl_setopt_array($http, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ['Content-Type: application/octet-stream'],
    CURLOPT_TIMEOUT => 60,
    // A shell that exports http_proxy would otherwise send these loopback calls to
    // that proxy. The request line then carries an absolute URI and the measurement
    // stops describing a direct call.
    CURLOPT_NOPROXY => '*',
]);

/** One request over a kept-alive connection. */
$post = static function (string $path, string $body) use ($http, $base): string {
    curl_setopt($http, CURLOPT_URL, $base . $path);
    curl_setopt($http, CURLOPT_POSTFIELDS, $body);
    $out = curl_exec($http);
    if (!is_string($out)) {
        throw new RuntimeException('request failed: ' . curl_error($http));
    }
    return $out;
};

// ---------------------------------------------------------------------- sessions

$options = (new OnnxOptions())
    ->intraOpThreads(THREADS)
    ->interOpThreads(1)
    ->parallel(false)
    ->optimization(OptimizationLevel::All);

$tiny = new OnnxSession($models . '/tiny.onnx', $options);
$encoder = new OnnxSession($models . '/encoder.onnx', $options);
$bert = new OnnxSession($models . '/bert.onnx', $options);
$classifier = new OnnxSession($models . '/classifier.onnx', $options);

$small = [1.0, 2.0, 3.0];
$steps = array_fill(0, 64, array_fill(0, 256, 0.01));
$tokens = array_fill(0, 128, array_fill(0, 768, 0.01));
$vector = array_fill(0, 256, 0.01);

// Packed forms, so neither side pays for a conversion the other does not.
$stepsBytes = OnnxValue::tensor($steps, OnnxValue::FLOAT32)->toBytes();
$tokensBytes = OnnxValue::tensor($tokens, OnnxValue::FLOAT32)->toBytes();
$vectorBytes = OnnxValue::tensor($vector, OnnxValue::FLOAT32)->toBytes();
$smallBytes = pack('f3', ...$small);

// ---------------------------------------------------------------------- the run

echo 'PHP ', PHP_VERSION, ', ONNX Runtime ', Build::version(), ', ', php_uname('m'), "\n";
echo 'intra_op=', THREADS, ' inter_op=1 sequential, ', ITERATIONS, " iterations after ", WARMUP, " warmup\n";

section('tiny.onnx  1x3  -- fixed cost of one call, no math');
row('extension  run()', measure(static fn () => $tiny->run(['x' => $small])['y']->toBytes()));
row('python over HTTP', measure(static fn () => $post('/run/tiny', $smallBytes)));

section('encoder.onnx  1x64x256  -- 6 x MatMul(256), 16384 floats in and out');
row('extension  array feed -> toBytes()', measure(static fn () => $encoder->run(['x' => $steps])['y']->toBytes()));
$packed = OnnxValue::fromBytes($stepsBytes);
row('extension  fromBytes feed -> toBytes()', measure(static fn () => $encoder->run(['x' => $packed])['y']->toBytes()));
row('python over HTTP  packed body -> raw', measure(static fn () => $post('/run/encoder', $stepsBytes)));
row("PHP pack('f*') of the same 16384 floats", measure(static fn () => pack('f*', ...array_merge(...$steps))));

section('bert.onnx  1x128x768  -- 12 x MatMul(768), the arithmetic of BERT-base');
row('extension  array feed -> toBytes()', measure(static fn () => $bert->run(['x' => $tokens])['y']->toBytes(), 10, 100));
$packedBig = OnnxValue::fromBytes($tokensBytes);
row('extension  fromBytes feed -> toBytes()', measure(static fn () => $bert->run(['x' => $packedBig])['y']->toBytes(), 10, 100));
row('python over HTTP  packed body -> raw', measure(static fn () => $post('/run/bert', $tokensBytes), 10, 100));

section('classifier.onnx  1x256 -> 1x1000  -- 1000 floats to move back');
row('extension  values()  (1000 PHP floats)', measure(static fn () => $classifier->run(['x' => $vector])['logits']->values()));
row('extension  toBytes()  (4 kB)', measure(static fn () => $classifier->run(['x' => $vector])['logits']->toBytes()));
row('extension  values() then a PHP argmax loop', measure(static function () use ($classifier, $vector): void {
    $scores = $classifier->run(['x' => $vector])['logits']->values();
    $best = 0;
    foreach ($scores as $i => $score) {
        if ($score > $scores[$best]) {
            $best = $i;
        }
    }
}));
row("extension  ['index' => 'max']", measure(static fn () => $classifier->run(['x' => $vector], ['logits' => ['index' => 'max']])['logits']->value()));
row('python over HTTP  argmax on its side', measure(static fn () => $post('/argmax/classifier', $vectorBytes)));

echo "\n";

proc_terminate($server);
proc_close($server);
