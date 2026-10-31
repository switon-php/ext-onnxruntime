<?php

declare(strict_types=1);

use Switon\OnnxRuntime\LogLevel;
use Switon\OnnxRuntime\OnnxException;
use Switon\OnnxRuntime\OnnxOptions;
use Switon\OnnxRuntime\Build;
use Switon\OnnxRuntime\OnnxSession;
use Switon\OnnxRuntime\OnnxValue;
use Switon\OnnxRuntime\OptimizationLevel;

$failures = 0;

function check(bool $ok, string $what): void
{
    global $failures;
    if (!$ok) {
        fwrite(STDERR, "FAIL {$what}\n");
        $failures++;
    }
}

function throws(callable $fn): ?OnnxException
{
    try {
        $fn();
    } catch (OnnxException $e) {
        return $e;
    }
    return null;
}

$dir = __DIR__ . '/models';
$identity = $dir . '/identity.onnx';

check(str_starts_with(Build::version(), '1.'), 'version');
$providers = Build::availableProviders();
check(in_array('cpu', $providers, true), 'cpu is available');
check(!str_contains(implode(',', $providers), 'ExecutionProvider'), 'providers are short');

$options = (new OnnxOptions())
    ->intraOpThreads(2)
    ->interOpThreads(2)
    ->optimization(OptimizationLevel::All)
    ->parallel(true)
    ->logLevel(LogLevel::Warning)
    ->verbosity(0)
    ->providers(['cpu']);
check($options instanceof OnnxOptions, 'options fluent');

$session = new OnnxSession($identity, $options);
check($session->providers() === ['cpu'], 'session providers');
$options->providers(['coreml', 'cpu']);
check($session->providers() === ['cpu'], 'options snapshot');

$inputs = $session->inputs();
check($inputs[0]->name() === 'x', 'input name');
check($inputs[0]->kind() === OnnxValue::TENSOR, 'input kind');
check($inputs[0]->type() === 'float32', 'input type');
check($inputs[0]->shape() === ['batch', 3], 'named dynamic shape');
$outputs = $session->outputs();
check($outputs[0]->name() === 'y' && $outputs[0]->type() === 'float32', 'output info');

$meta = $session->metadata();
check($meta->producer() === 'test', 'producer');
check($meta->graphName() === 'g', 'graph name');
check($meta->domain() === 'test.domain', 'domain');
check($meta->description() === 'identity', 'description');
check($meta->version() === 3, 'model version');
check($meta->custom() === ['labels' => 'cat,dog'], 'custom metadata');

$promoted = $session->run(['x' => [1.0, 2.0, 3.0]]);
$hidden = $promoted['y'];
check($hidden->kind() === OnnxValue::TENSOR, 'result kind');
check($hidden->elementType() === OnnxValue::FLOAT32, 'result type');
check($hidden->shape() === [1, 3], 'dynamic batch of one');
check($hidden->values() === [1.0, 2.0, 3.0], 'flat scores');
$bytes = $hidden->toBytes();
check(strlen($bytes) === 12, 'toBytes length');
unset($session);
check(strlen($hidden->toBytes()) === 12, 'value outlives session');

$session = new OnnxSession($identity);
$batch = $session->run(['x' => [[1, 2, 3], [4, 5, 6]]])['y'];
check($batch->shape() === [2, 3], 'explicit batch');
check($batch->values()[3] === 4.0, 'second row');

$packed = OnnxValue::fromBytes($bytes);
check($packed->kind() === OnnxValue::TENSOR, 'fromBytes kind');
check($packed->toBytes() === $bytes, 'fromBytes roundtrip');
$again = $session->run(['x' => $packed])['y'];
check($again->shape() === [1, 3] && $again->values() === [1.0, 2.0, 3.0], 'run packed bytes');
$fromBytesType = throws(fn () => $packed->elementType());
check($fromBytesType !== null, 'fromBytes type throws');

$typed = OnnxValue::tensor([10, 20, 30], OnnxValue::INT32);
check($typed->elementType() === 'int32' && $typed->shape() === [3], 'typed tensor');
$reused = OnnxValue::tensor([10, 20, 30]);
check($reused->elementType() === OnnxValue::INT64, 'ints infer int64');
$ints = new OnnxSession($dir . '/ints.onnx');
check($ints->run(['x' => $reused])['y']->values() === [10, 20, 30], 'reuse typed value');
check($ints->run(['x' => $reused])['y']->shape() === [3], 'second run');
$mixed = OnnxValue::tensor([1, 2.5]);
check($mixed->elementType() === OnnxValue::FLOAT32 && $mixed->values() === [1.0, 2.5], 'int float mix');
check(throws(fn () => OnnxValue::tensor([])) !== null, 'empty tensor throws');
check(throws(fn () => OnnxValue::tensor([1], OnnxValue::FLOAT16)) !== null, 'float16 tensor throws');
check(OnnxValue::tensor([], OnnxValue::FLOAT32)->shape() === [0], 'empty typed tensor');

$strings = new OnnxSession($dir . '/strings.onnx');
$words = $strings->run(['x' => OnnxValue::tensor(['a', 'b'])])['y'];
check($words->elementType() === OnnxValue::STRING && $words->values() === ['a', 'b'], 'string tensor');
$stringFeed = throws(fn () => $strings->run(['x' => 'ab']));
check($stringFeed !== null && str_contains($stringFeed->getMessage(), 'A PHP string is packed bytes'), 'php string message');
check($stringFeed->inputName() === 'x', 'string feed input');

$fixed = new OnnxSession($dir . '/fixed.onnx');
$mismatch = throws(fn () => $fixed->run(['x' => [1.0, 2.0, 3.0]]));
check($mismatch !== null && $mismatch->inputName() === 'x', 'fixed dim input');
check($mismatch->expectedShape() === [2, 3], 'expected shape');
check($mismatch->actualShape() === [3], 'actual shape');
$missing = throws(fn () => $fixed->run([]));
check($missing !== null && $missing->inputName() === 'x' && $missing->actualShape() === null, 'required input');
check($missing->expectedShape() === [2, 3], 'required expected shape');

$unnamed = new OnnxSession($dir . '/unnamed.onnx');
check($unnamed->inputs()[0]->shape() === [null, 4], 'unnamed dynamic dim');
check($unnamed->run(['x' => [1, 2, 3, 4]])['y']->shape() === [1, 4], 'unnamed promotion');

$two = new OnnxSession($dir . '/two.onnx');
$all = $two->run(['x' => [1.0, 2.0, 3.0]]);
check(array_keys($all) === ['y', 'z'], 'null fetches graph order');
check(array_keys($two->run(['x' => [1.0, 2.0, 3.0]], [])) === ['y', 'z'], 'empty list fetches all');
check(array_keys($two->run(['x' => [1.0, 2.0, 3.0]], ['z', 'y'])) === ['z', 'y'], 'output order');

$add = new OnnxSession($dir . '/add.onnx');
check($add->run(['x' => [1.0, 2.0, 3.0]])['y']->values() === [2.0, 3.0, 4.0], 'initializer may be omitted');

$loaded = OnnxSession::fromContents(file_get_contents($identity));
check($loaded->run(['x' => [4.0, 5.0, 6.0]])['y']->values() === [4.0, 5.0, 6.0], 'session fromContents');

$seq = (new OnnxSession($dir . '/seq.onnx'))->run(['x' => [1.0, 2.0, 3.0]])['seq'];
check($seq->kind() === OnnxValue::SEQUENCE, 'sequence kind');
$items = $seq->items();
check(count($items) === 1 && $items[0]->values() === [1.0, 2.0, 3.0], 'sequence items');
check(throws(fn () => $items[0]->items()) !== null, 'items rejects a tensor');

$zip = (new OnnxSession($dir . '/zipmap.onnx'))->run(['x' => [[0.25, 0.75]]])['scores'];
check($zip->kind() === OnnxValue::SEQUENCE, 'zipmap sequence');
$entries = $zip->items()[0]->entries();
check($entries[0][0] === 'no' && $entries[1][0] === 'yes', 'map keys');
check($entries[1][1]->values()[0] === 0.75, 'map score');

$f16 = pack('v2', 0x3C00, 0x4000);
$half = (new OnnxSession($dir . '/f16.onnx'))->run(['x' => OnnxValue::fromBytes($f16)])['y'];
check($half->elementType() === OnnxValue::FLOAT16, 'float16 type');
check($half->values() === [1.0, 2.0], 'float16 widens');
check($half->toBytes() === $f16, 'float16 bytes');

$twoDyn = throws(fn () => (new OnnxSession($dir . '/twodyn.onnx'))->run(['x' => pack('f*', 1.0, 2.0, 3.0, 4.0)]));
check($twoDyn !== null && str_contains($twoDyn->getMessage(), 'two or more dynamic dimensions'), 'two dynamic dims');
check($twoDyn?->inputName() === 'x' && $twoDyn?->expectedShape() === ['batch', 'seq'] && $twoDyn?->actualShape() === null, 'two dynamic shapes');

$fed = (new OnnxSession($dir . '/seq_at.onnx'))->run(['seq' => $seq])['y'];
check($fed->values() === [1.0, 2.0, 3.0], 'feed a sequence back');

$negative = throws(fn () => new OnnxSession($identity, (new OnnxOptions())->intraOpThreads(-1)));
check($negative !== null && $negative->inputName() === null, 'negative threads');
$empty = throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers([])));
check($empty !== null, 'empty providers');

if ($failures > 0) {
    fwrite(STDERR, "{$failures} failed\n");
    exit(1);
}
echo "ok\n";
