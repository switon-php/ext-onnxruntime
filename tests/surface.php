<?php

declare(strict_types=1);

use Switon\OnnxRuntime\LogLevel;
use Switon\OnnxRuntime\OnnxException;
use Switon\OnnxRuntime\OnnxValueInfo;
use Switon\OnnxRuntime\OnnxMetadata;
use Switon\OnnxRuntime\OnnxOptions;
use Switon\OnnxRuntime\Build;
use Switon\OnnxRuntime\OnnxSession;
use Switon\OnnxRuntime\OnnxValue;
use Switon\OnnxRuntime\OptimizationLevel;

$failures = 0;
$called = [];

function check(bool $ok, string $what): void
{
    global $failures;
    if (!$ok) {
        fwrite(STDERR, "FAIL {$what}\n");
        $failures++;
    }
}

function saw(string $class, string $method): void
{
    global $called;
    $called[$class . '::' . $method] = true;
}

$dir = __DIR__ . '/models';

saw(Build::class, 'version');
check(str_starts_with(Build::version(), '1.'), 'runtime version');
saw(Build::class, 'availableProviders');
check(in_array('cpu', Build::availableProviders(), true), 'runtime providers');

$options = new OnnxOptions();
saw(OnnxOptions::class, '__construct');
saw(OnnxOptions::class, 'intraOpThreads');
saw(OnnxOptions::class, 'interOpThreads');
saw(OnnxOptions::class, 'parallel');
saw(OnnxOptions::class, 'verbosity');
saw(OnnxOptions::class, 'providers');
saw(OnnxOptions::class, 'freeDimensions');
$options->intraOpThreads(1)->interOpThreads(1)->parallel(false)->verbosity(0)->providers(['cpu']);
check((new OnnxOptions())->freeDimensions(['batch' => 1]) instanceof OnnxOptions, 'free dimensions');
saw(OptimizationLevel::class, 'cases');
saw(OptimizationLevel::class, 'from');
saw(OptimizationLevel::class, 'tryFrom');
check(OptimizationLevel::from('all') === OptimizationLevel::All && OptimizationLevel::tryFrom('nope') === null, 'optimization enum');
foreach (OptimizationLevel::cases() as $level) {
    saw(OptimizationLevel::class, $level->name);
    saw(OnnxOptions::class, 'optimization');
    $options->optimization($level);
}
saw(LogLevel::class, 'cases');
saw(LogLevel::class, 'from');
saw(LogLevel::class, 'tryFrom');
check(LogLevel::from('warning') === LogLevel::Warning && LogLevel::tryFrom('nope') === null, 'log level enum');
foreach (LogLevel::cases() as $level) {
    saw(LogLevel::class, $level->name);
    saw(OnnxOptions::class, 'logLevel');
    $options->logLevel($level);
}

$session = new OnnxSession($dir . '/identity.onnx', $options);
saw(OnnxSession::class, '__construct');
saw(OnnxSession::class, 'providers');
check($session->providers() === ['cpu'], 'session providers');

$input = $session->inputs()[0];
saw(OnnxSession::class, 'inputs');
check($input instanceof OnnxValueInfo, 'input info');
saw(OnnxValueInfo::class, 'name');
saw(OnnxValueInfo::class, 'kind');
saw(OnnxValueInfo::class, 'type');
saw(OnnxValueInfo::class, 'shape');
check($input->name() === 'x' && $input->kind() === 'tensor' && $input->type() === 'float32' && $input->shape() === ['batch', 3], 'input fields');

$output = $session->outputs()[0];
saw(OnnxSession::class, 'outputs');
check($output->name() === 'y', 'output name');

$meta = $session->metadata();
saw(OnnxSession::class, 'metadata');
check($meta instanceof OnnxMetadata, 'metadata');
saw(OnnxMetadata::class, 'producer');
saw(OnnxMetadata::class, 'graphName');
saw(OnnxMetadata::class, 'domain');
saw(OnnxMetadata::class, 'description');
saw(OnnxMetadata::class, 'version');
saw(OnnxMetadata::class, 'custom');
check($meta->producer() === 'test' && $meta->graphName() === 'g' && $meta->domain() === 'test.domain', 'metadata text');
check($meta->description() === 'identity' && $meta->version() === 3 && $meta->custom() === ['labels' => 'cat,dog'], 'metadata rest');

$result = $session->run(['x' => [1.0, 2.0, 3.0]])['y'];
saw(OnnxSession::class, 'run');
saw(OnnxValue::class, 'kind');
saw(OnnxValue::class, 'elementType');
saw(OnnxValue::class, 'shape');
saw(OnnxValue::class, 'values');
saw(OnnxValue::class, 'toBytes');
check($result->kind() === 'tensor' && $result->elementType() === 'float32' && $result->shape() === [1, 3], 'result tensor');
check($result->values() === [1.0, 2.0, 3.0] && strlen($result->toBytes()) === 12, 'result values');

$one = OnnxValue::tensor([7]);
saw(OnnxValue::class, 'tensor');
saw(OnnxValue::class, 'value');
check($one->value() === 7, 'one element');

$packed = OnnxValue::fromBytes($result->toBytes());
saw(OnnxValue::class, 'fromBytes');
check($packed->toBytes() === $result->toBytes(), 'value fromBytes');

$loaded = OnnxSession::fromContents((string) file_get_contents($dir . '/identity.onnx'));
saw(OnnxSession::class, 'fromContents');
check($loaded->run(['x' => [4.0, 5.0, 6.0]])['y']->values() === [4.0, 5.0, 6.0], 'session fromContents');

$seq = (new OnnxSession($dir . '/seq.onnx'))->run(['x' => [1.0, 2.0, 3.0]])['seq'];
saw(OnnxValue::class, 'items');
check($seq->items()[0]->values() === [1.0, 2.0, 3.0], 'sequence items');

$map = (new OnnxSession($dir . '/zipmap.onnx'))->run(['x' => [[0.25, 0.75]]])['scores']->items()[0];
saw(OnnxValue::class, 'entries');
check($map->entries()[1][0] === 'yes' && $map->entries()[1][1]->value() === 0.75, 'map entries');

saw(OnnxValue::class, '__debugInfo');
$debug = $result->__debugInfo();
check($debug['kind'] === 'tensor' && $debug['type'] === 'float32' && $debug['shape'] === [1, 3], 'debug info');

saw(OnnxValue::class, '__serialize');
saw(OnnxValue::class, '__unserialize');
$restored = unserialize(serialize($result));
check($restored->values() === [1.0, 2.0, 3.0] && $restored->shape() === [1, 3] && $restored->elementType() === 'float32', 'value round trip');

saw(OnnxValue::class, 'count');
check(count($result) === 3 && count($seq) === 1 && count($map) === 2, 'value count');

saw(OnnxValue::class, 'getIterator');
$iterated = 0;
foreach ($seq as $item) {
    $iterated++;
    check($item instanceof OnnxValue, 'iterated item');
}
check($iterated === 1, 'sequence iteration');

saw(OnnxValue::class, 'jsonSerialize');
check(json_encode($result) === '[[1,2,3]]' && json_encode($map) === '{"no":[0.25],"yes":[0.75]}', 'value json');

saw(OnnxSession::class, 'close');
$closing = new OnnxSession($dir . '/identity.onnx');
$kept = $closing->run(['x' => [1.0, 2.0, 3.0]])['y'];
$closing->close();
check($closing->providers() === ['cpu'] && $kept->values() === [1.0, 2.0, 3.0], 'close keeps providers and values');

try {
    (new OnnxSession($dir . '/fixed.onnx'))->run(['x' => [1.0, 2.0, 3.0]]);
    check(false, 'shape mismatch throws');
} catch (OnnxException $mismatch) {
    saw(OnnxException::class, 'inputName');
    saw(OnnxException::class, 'expectedShape');
    saw(OnnxException::class, 'actualShape');
    check($mismatch->inputName() === 'x' && $mismatch->expectedShape() === [2, 3] && $mismatch->actualShape() === [3], 'exception shapes');
}

$namespace = 'Switon\\OnnxRuntime\\';
$found = 0;
foreach (get_declared_classes() as $class) {
    if (!str_starts_with($class, $namespace)) {
        continue;
    }
    $found++;
    $ref = new ReflectionClass($class);
    if ($ref->isEnum()) {
        foreach ((new ReflectionEnum($class))->getCases() as $case) {
            check(isset($called[$class . '::' . $case->getName()]), 'case ' . $class . '::' . $case->getName());
        }
    }
    foreach ($ref->getMethods(ReflectionMethod::IS_PUBLIC) as $method) {
        if ($method->getDeclaringClass()->getName() !== $class) {
            continue;
        }
        check(isset($called[$class . '::' . $method->getName()]), 'method ' . $class . '::' . $method->getName());
    }
}
check($found === 9, 'nine classes');

if ($failures > 0) {
    fwrite(STDERR, "{$failures} failed\n");
    exit(1);
}
echo "ok\n";
