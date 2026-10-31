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

function message(?OnnxException $e): string
{
    return $e?->getMessage() ?? '';
}

$dir = __DIR__ . '/models';
$identity = $dir . '/identity.onnx';

ob_start();
phpinfo(INFO_MODULES);
check(str_contains((string) ob_get_clean(), 'onnxruntime'), 'phpinfo');

$twice = (new OnnxOptions())
    ->optimization(OptimizationLevel::All)
    ->optimization(OptimizationLevel::Off)
    ->logLevel(LogLevel::Warning)
    ->logLevel(LogLevel::Verbose)
    ->verbosity(1)
    ->intraOpThreads(0)
    ->parallel(false);
check($twice instanceof OnnxOptions, 'set options twice');
unset($twice);

foreach ([OptimizationLevel::Off, OptimizationLevel::Basic, OptimizationLevel::Extended, OptimizationLevel::All] as $level) {
    $session = new OnnxSession($identity, (new OnnxOptions())->optimization($level));
    check($session->run(['x' => [1.0, 2.0, 3.0]])['y']->values() === [1.0, 2.0, 3.0], 'optimization ' . $level->value);
}
foreach ([LogLevel::Verbose, LogLevel::Info, LogLevel::Error, LogLevel::Fatal, LogLevel::Warning] as $level) {
    $session = new OnnxSession($identity, (new OnnxOptions())->logLevel($level)->verbosity(0));
    check($session->providers() === ['cpu'], 'log ' . $level->value);
}

$full = new OnnxSession($identity, (new OnnxOptions())->providers(['CPUExecutionProvider']));
check($full->providers() === ['cpu'], 'provider full name');
if (in_array('coreml', Build::availableProviders(), true)) {
    $core = new OnnxSession($identity, (new OnnxOptions())->providers(['coreml', 'cpu']));
    check($core->providers() === ['coreml', 'cpu'], 'coreml then cpu');
}
foreach (Build::availableProviders() as $providerName) {
    $failed = throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers([$providerName])));
    check($failed === null || !str_contains(message($failed), 'unknown provider'), "short name {$providerName}");
    if ($failed !== null) {
        continue;
    }
    $listed = new OnnxSession($identity, (new OnnxOptions())->providers([$providerName]));
    check($listed->providers() === [$providerName], "short name {$providerName} registered");
    $again = new OnnxSession($identity, (new OnnxOptions())->providers($listed->providers()));
    check($again->providers() === [$providerName], "short name {$providerName} round trip");
}
$withDevice = new OnnxSession($identity, (new OnnxOptions())->providers([['cpu', ['device_id' => '0']]]));
check($withDevice->providers() === ['cpu'], 'provider option');

check(str_contains(message(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers(['cuda'])))), 'provider cuda'), 'cuda missing');
check(str_contains(message(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers(['tensorrt'])))), 'provider tensorrt'), 'tensorrt missing');
check(str_contains(message(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers(['nope'])))), 'unknown provider'), 'unknown provider');
check(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers(['cpu' => 'x']))) !== null, 'providers not a list');
check(str_contains(message(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers([['cpu', ['device_id' => 0]]])))), 'map of strings'), 'option value not a string');
check(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers([['cpu']]))) !== null, 'provider pair too short');
check(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers([1]))) !== null, 'provider entry not a name');
check(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->interOpThreads(-1))) !== null, 'negative parallel threads');
check(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->verbosity(-1))) !== null, 'negative verbosity');
check(str_contains(message(throws(fn () => new OnnxSession($dir . '/missing.onnx'))), 'failed to load'), 'missing file');
check(str_contains(message(throws(fn () => OnnxSession::fromContents('not-a-model'))), 'failed to load model contents'), 'bad model contents');

$named = throws(fn () => (new OnnxSession($identity))->run(['x' => [1.0, 2.0]]));
check($named !== null && str_contains(message($named), 'batch'), 'named dim in the message');
$unnamed = new OnnxSession($dir . '/unnamed.onnx');
$nullDim = throws(fn () => $unnamed->run(['x' => [[1.0, 2.0, 3.0]]]));
check($nullDim !== null && str_contains(message($nullDim), 'null'), 'unnamed dim in the message');

$one = new OnnxSession($dir . '/onedyn.onnx');
$solved = $one->run(['x' => pack('f*', 1.0, 2.0, 3.0)])['y'];
check($solved->shape() === [3] && $solved->values() === [1.0, 2.0, 3.0], 'one dynamic dim from bytes');
$badLen = throws(fn () => $one->run(['x' => 'abc']));
check($badLen !== null && str_contains(message($badLen), 'packed length'), 'one dynamic length');
$fixedLen = throws(fn () => (new OnnxSession($dir . '/f16.onnx'))->run(['x' => pack('v', 0x3C00)]));
check($fixedLen !== null && str_contains(message($fixedLen), 'packed length'), 'fixed packed length');

$special = (new OnnxSession($dir . '/f16.onnx'))->run(['x' => pack('v*', 0x0000, 0x0001)])['y']->values();
check($special[0] === 0.0 && $special[1] > 0.0 && $special[1] < 1e-4, 'float16 subnormal');
$ends = (new OnnxSession($dir . '/f16.onnx'))->run(['x' => pack('v*', 0x7C00, 0x7E00)])['y']->values();
check($ends[0] === INF && is_nan($ends[1]), 'float16 inf and nan');
$neg = (new OnnxSession($dir . '/f16.onnx'))->run(['x' => pack('v*', 0x8000, 0xBC00)])['y']->values();
check($neg[0] === 0.0 && $neg[1] === -1.0, 'float16 negative');

$bf = (new OnnxSession($dir . '/bf16.onnx'))->run(['x' => pack('v*', 0x3F80, 0x4000)])['y'];
check($bf->elementType() === 'bfloat16' && $bf->values() === [1.0, 2.0], 'bfloat16 widens');
check(str_contains(message(throws(fn () => (new OnnxSession($dir . '/bf16.onnx'))->run(['x' => [1.0, 2.0]]))), 'fromBytes'), 'bfloat16 array');

$i8 = (new OnnxSession($dir . '/i8.onnx'))->run(['x' => [1, -2]])['y'];
check($i8->elementType() === 'int8' && $i8->values() === [1, -2], 'int8');
check(str_contains(message(throws(fn () => (new OnnxSession($dir . '/i8.onnx'))->run(['x' => [200, 0]]))), 'outside'), 'int8 range');
$u8 = (new OnnxSession($dir . '/u8.onnx'))->run(['x' => [1, 255]])['y'];
check($u8->elementType() === 'uint8' && $u8->values() === [1, 255], 'uint8');
$i16 = (new OnnxSession($dir . '/i16.onnx'))->run(['x' => [1, -2]])['y'];
check($i16->elementType() === 'int16' && $i16->values() === [1, -2], 'int16');
$u16 = (new OnnxSession($dir . '/u16.onnx'))->run(['x' => [1, 65535]])['y'];
check($u16->elementType() === 'uint16' && $u16->values() === [1, 65535], 'uint16');
$u32 = (new OnnxSession($dir . '/u32.onnx'))->run(['x' => [1, 4000000000]])['y'];
check($u32->elementType() === 'uint32' && $u32->values() === [1, 4000000000], 'uint32');
$u64 = (new OnnxSession($dir . '/u64.onnx'))->run(['x' => pack('Q*', 1, 2)])['y'];
check($u64->elementType() === 'uint64', 'uint64');
check(throws(fn () => $u64->values()) !== null, 'uint64 values');
check(str_contains(message(throws(fn () => (new OnnxSession($dir . '/u64.onnx'))->run(['x' => [1, 2]]))), 'fromBytes'), 'uint64 array');
$i32 = (new OnnxSession($dir . '/i32.onnx'))->run(['x' => [1.0, 2.0]])['y'];
check($i32->values() === [1, 2], 'float whole number into int32');
$wide = (new OnnxSession($dir . '/f64.onnx'))->run(['x' => [1, 2.5]])['y'];
check($wide->elementType() === 'float64' && $wide->values() === [1.0, 2.5], 'float64');
$flags = (new OnnxSession($dir . '/bools.onnx'))->run(['x' => [true, 0]])['y'];
check($flags->elementType() === 'bool' && $flags->values() === [true, false], 'bool tensor');

$typed = OnnxValue::tensor([1.5, 2], OnnxValue::FLOAT32);
check($typed->values() === [1.5, 2.0], 'float64 name is not required');
$doubles = OnnxValue::tensor([1, 2.5], 'float64');
check($doubles->elementType() === 'float64' && $doubles->toBytes() !== '', 'float64 tensor');
$bools = OnnxValue::tensor([true, false]);
check($bools->elementType() === OnnxValue::BOOL && $bools->values() === [true, false], 'inferred bool');
$bits = OnnxValue::tensor([0, 1], OnnxValue::BOOL);
check($bits->values() === [false, true], 'bool from 0 and 1');
check(OnnxValue::tensor([], OnnxValue::INT64)->toBytes() === '', 'empty tensor bytes');
check(str_contains(message(throws(fn () => OnnxValue::tensor([1], 'nope'))), 'unknown element type'), 'unknown type');
check(str_contains(message(throws(fn () => OnnxValue::tensor([true, 1]))), 'cannot be mixed'), 'bool mixed with int');
check(str_contains(message(throws(fn () => OnnxValue::tensor(['a', 1]))), 'cannot be mixed'), 'string mixed with int');
check(str_contains(message(throws(fn () => OnnxValue::tensor(['a' => 1]))), 'must be a list'), 'assoc array');
check(str_contains(message(throws(fn () => OnnxValue::tensor([[1, 2], [3]]))), 'same length'), 'ragged rows');
check(str_contains(message(throws(fn () => OnnxValue::tensor([[[1]], [[2, 3]]]))), 'same length'), 'ragged rank');
$deep = [1];
for ($i = 0; $i < 8; $i++) {
    $deep = [$deep];
}
check(throws(fn () => OnnxValue::tensor($deep)) !== null, 'rank above 8');
check(str_contains(message(throws(fn () => OnnxValue::tensor([1.5], OnnxValue::INT32))), 'not an integer'), 'fraction into int');
check(str_contains(message(throws(fn () => OnnxValue::tensor([INF], OnnxValue::INT32))), 'does not fit'), 'inf into int');
check(str_contains(message(throws(fn () => OnnxValue::tensor([200], OnnxValue::INT8))), 'outside'), 'int8 overflow');
check(str_contains(message(throws(fn () => OnnxValue::tensor([-1], OnnxValue::UINT8))), 'outside'), 'uint8 overflow');
check(str_contains(message(throws(fn () => OnnxValue::tensor([2], OnnxValue::BOOL))), 'true, false, 0, or 1'), 'bad bool');
check(str_contains(message(throws(fn () => OnnxValue::tensor(['a'], OnnxValue::INT32))), 'expected a number'), 'string into int');
check(str_contains(message(throws(fn () => OnnxValue::tensor([16777217], OnnxValue::FLOAT32))), 'does not fit in float32'), 'int wider than float32');
check(str_contains(message(throws(fn () => (new OnnxSession($identity))->run(['x' => ['a', 'b', 'c']]))), 'expected a number'), 'string into float feed');

$strings = new OnnxSession($dir . '/strings.onnx');
check(str_contains(message(throws(fn () => $strings->run(['x' => OnnxValue::fromBytes('ab')]))), 'packed bytes'), 'fromBytes into a string tensor');
$words = $strings->run(['x' => ['a', 'b']])['y'];
check(throws(fn () => $words->toBytes()) !== null, 'string tensor toBytes');
check(str_contains(message(throws(fn () => $strings->run(['x' => [1, 2]]))), 'string elements'), 'number into a string tensor');
check($strings->run(['x' => ["ab\0cd", 'e']])['y']->values()[0] === "ab\0cd", 'NUL inside a string element');
check($strings->run(['x' => ["\0\0", '']])['y']->values() === ["\0\0", ''], 'all-NUL and empty string elements');
check($strings->run(['x' => ["\xff\xfe", 'e']])['y']->values()[0] === "\xff\xfe", 'non-UTF8 string element');

$seqSession = new OnnxSession($dir . '/seq.onnx');
check($seqSession->outputs()[0]->type() === 'sequence<float32>', 'sequence declaration');
$seq = $seqSession->run(['x' => [1.0, 2.0, 3.0]])['seq'];
check(throws(fn () => $seq->toBytes()) !== null, 'sequence toBytes');
check(throws(fn () => $seq->values()) !== null, 'sequence values');
check(throws(fn () => $seq->elementType()) !== null, 'sequence type');
check(throws(fn () => $seq->shape()) !== null, 'sequence shape');
check(throws(fn () => $seq->entries()) !== null, 'sequence entries');

// Elements of a sequence do not have to share a shape: that is what a sequence is for.
$ragged = (new OnnxSession($dir . '/ragged.onnx'))->run([])['ragged'];
check($ragged->count() === 3, 'ragged sequence count');
check(array_map(static fn ($item) => $item->shape(), $ragged->items()) === [[1], [2], [3]], 'ragged sequence shapes');
check(array_map(static fn ($item) => $item->values(), $ragged->items()) === [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]], 'ragged sequence values');
check(json_encode($ragged) === '[[1],[2,3],[4,5,6]]', 'ragged sequence json');
check(str_contains(message(throws(fn () => (new OnnxSession($identity))->run(['x' => $seq]))), 'sequence'), 'sequence fed to a tensor');
check(str_contains(message(throws(fn () => (new OnnxSession($dir . '/seq_at.onnx'))->run(['seq' => [1.0, 2.0, 3.0]]))), 'sequence'), 'array fed to a sequence');

$zip = (new OnnxSession($dir . '/zipmap.onnx'))->run(['x' => [[0.25, 0.75]]])['scores'];
$map = $zip->items()[0];
check($map->kind() === OnnxValue::MAP, 'map kind');
check((new OnnxSession($dir . '/zipmap.onnx'))->outputs()[0]->type() === 'sequence<map<string,float32>>', 'map declaration');
check(throws(fn () => $map->items()) !== null, 'map items');
check(throws(fn () => $map->toBytes()) !== null, 'map toBytes');
check(throws(fn () => $map->values()) !== null, 'map values');
check(throws(fn () => $map->elementType()) !== null, 'map type');
check(str_contains(message(throws(fn () => (new OnnxSession($identity))->run(['x' => $map]))), 'map'), 'map fed to a tensor');
$intMapValue = (new OnnxSession($dir . '/zipint.onnx'))->run(['x' => [[0.25, 0.75]]])['scores']->items()[0];
$intMap = $intMapValue->entries();
check($intMap[0][0] === 10 && $intMap[1][1]->values()[0] === 0.75, 'int map keys');
$cast = new OnnxSession($dir . '/castmap.onnx');
check($cast->inputs()[0]->kind() === OnnxValue::MAP && $cast->inputs()[0]->type() === 'map<int64,float32>', 'map input declaration');
check($cast->run(['x' => $intMapValue])['y']->values() === [0.25, 0.75], 'feed a map');
$tooBig = throws(fn () => (new OnnxSession($dir . '/huge.onnx'))->run(['x' => OnnxValue::fromBytes('abcd')]));
check($tooBig !== null && str_contains(message($tooBig), 'too large'), 'shape too large');

$optional = new OnnxSession($dir . '/optional.onnx');
check($optional->inputs()[0]->type() === 'float32', 'optional unwraps');
check($optional->run(['x' => [1.0, 2.0, 3.0]])['y']->values() === [true], 'optional supplied');
check($optional->run([])['y']->values() === [false], 'optional omitted');

$bare = (new OnnxSession($dir . '/bare.onnx'))->metadata();
check($bare->producer() === '' && $bare->domain() === '' && $bare->description() === '' && $bare->version() === 0, 'omitted metadata');
check((new OnnxSession($dir . '/dup.onnx'))->metadata()->custom()['labels'] === 'second', 'duplicate metadata key');

$ran = throws(fn () => (new OnnxSession($dir . '/gather.onnx'))->run(['x' => [1.0, 2.0, 3.0], 'i' => [9]]));
check($ran !== null, 'runtime failure');

$session = new OnnxSession($identity);
check(str_contains(message(throws(fn () => $session->run([1.0, 2.0, 3.0]))), 'feed key'), 'integer feed key');
check(str_contains(message(throws(fn () => $session->run(['nope' => [1.0, 2.0, 3.0]]))), 'unknown input'), 'unknown input');
check(str_contains(message(throws(fn () => $session->run(['x' => [1.0, 2.0, 3.0], 'x ' => [1.0, 2.0, 3.0]]))), 'unknown input'), 'near name');
$repeat = ['x' => [1.0, 2.0, 3.0]];
$repeat['x'] = [4.0, 5.0, 6.0];
check($session->run($repeat)['y']->values() === [4.0, 5.0, 6.0], 'php array keeps one key');
check(throws(fn () => $session->run(['x' => 1])) !== null, 'feed is a number');
check(throws(fn () => $session->run(['x' => [1.0, 2.0, 3.0]], ['y' => 0])) !== null, 'output names not a list');
check(throws(fn () => $session->run(['x' => [1.0, 2.0, 3.0]], [1])) !== null, 'output name not a string');
check(throws(fn () => $session->run(['x' => [1.0, 2.0, 3.0]], ['y', 'y'])) !== null, 'repeated output');
check(throws(fn () => $session->run(['x' => [1.0, 2.0, 3.0]], ['missing'])) !== null, 'unknown output');

foreach ([
    'f8' => 'float8e4m3fn',
    'f8fnuz' => 'float8e4m3fnuz',
    'f8e5' => 'float8e5m2',
    'f8e5fnuz' => 'float8e5m2fnuz',
    'i4' => 'unknown',
] as $file => $type) {
    $out = (new OnnxSession($dir . '/' . $file . '.onnx'))->run(['x' => [1.0, 2.0]])['y'];
    check($out->elementType() === $type, $type);
    check(throws(fn () => $out->values()) !== null, $type . ' values');
}

check(OnnxValue::tensor(['a', 'b'], OnnxValue::STRING)->values() === ['a', 'b'], 'string type name');
check(str_contains(message(throws(fn () => OnnxValue::tensor([null]))), 'ints, floats, bools, or strings'), 'null element');
check(str_contains(message(throws(fn () => OnnxValue::tensor(['a'], 'float64'))), 'expected a number'), 'string into float64');
check(str_contains(message(throws(fn () => OnnxValue::tensor([[1], [[2]]]))), 'same length'), 'leaf row is a list');
check(str_contains(message(throws(fn () => OnnxValue::tensor([[[1]], [2]]))), 'same length'), 'a row stops early');
$plain = throws(fn () => new OnnxSession($identity, (new OnnxOptions())->intraOpThreads(-1)));
check($plain?->expectedShape() === null && $plain?->actualShape() === null, 'exception shapes absent');
check(throws(fn () => new OnnxSession($identity, (new OnnxOptions())->providers([[1, []]]))) !== null, 'provider name is not a string');

foreach ([Build::class, OnnxValue::class, OnnxOptions::class] as $class) {
    $method = new ReflectionMethod($class, '__construct');
    $method->invoke((new ReflectionClass($class))->newInstanceWithoutConstructor());
}
$info = (new OnnxSession($identity))->inputs()[0];
(new ReflectionMethod($info, '__construct'))->invoke($info);
$meta = (new OnnxSession($identity))->metadata();
(new ReflectionMethod($meta, '__construct'))->invoke($meta);
$reflected = (new ReflectionMethod(OnnxSession::class, 'fromContents'))->invoke(null, (string) file_get_contents($identity));
check($reflected->run(['x' => [1.0, 2.0, 3.0]])['y']->values() === [1.0, 2.0, 3.0], 'fromContents without a called class');
check((new ReflectionMethod(OnnxValue::class, 'tensor'))->invoke(null, [4]) instanceof OnnxValue, 'tensor without a called class');

$mask = new OnnxSession($dir . '/maskio.onnx');
$ids = [4, 5, 6, 7];
$filled = $mask->run([
    'input_ids' => $ids,
    'encoder_attention_mask' => 1,
])['mask'];
check($filled->shape() === [1, 4] && $filled->values() === [1, 1, 1, 1], 'mask shorthand 1');
$right = $mask->run([
    'input_ids' => $ids,
    'encoder_attention_mask' => ['sequence' => [2]],
])['mask']->values();
check($right === [1, 1, 0, 0], 'mask sequence pads right');
$left = $mask->run([
    'input_ids' => $ids,
    'encoder_attention_mask' => ['sequence' => [2], 'left' => true],
])['mask']->values();
check($left === [0, 0, 1, 1], 'mask sequence pads left');
$batch = $mask->run([
    'input_ids' => array_fill(0, 3, array_fill(0, 8, 1)),
    'encoder_attention_mask' => ['sequence' => [5, 8, 3]],
])['mask']->values();
check($batch === [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 'mask sequence batch');
check(str_contains(message(throws(fn () => $mask->run(['input_ids' => $ids, 'encoder_attention_mask' => 2]))), 'shorthand is 1'), 'mask shorthand rejects 2');
check(throws(fn () => (new OnnxSession($identity))->run(['x' => 1])) !== null, 'shorthand only on attention_mask');

$scores = new OnnxSession($dir . '/logits.onnx');
$vectors = [[1.0, 3.0, 2.0], [0.5, 0.2, 9.0]];
$last = $scores->run(['scores' => [$vectors]], ['logits' => ['index' => 'max']])['logits'];
check($last->shape() === [1] && $last->value() === 2, 'index max is the last step');
$each = $scores->run(['scores' => [$vectors]], ['logits' => ['index' => 'max', 'step' => 'each']])['logits'];
check($each->shape() === [1, 2] && $each->values() === [1, 2], 'index max each step');
check(str_contains(message(throws(fn () => $each->value())), 'exactly one element'), 'value needs one element');
$whole = $scores->run(['scores' => [$vectors]], ['logits'])['logits'];
check($whole->elementType() === 'float32' && count($whole->values()) === 6, 'named output stays the tensor');
check(str_contains(message(throws(fn () => $scores->run(['scores' => [$vectors]], ['logits' => ['index' => 'min']]))), "index must be 'max'"), 'index only max');

$grid = OnnxValue::tensor([[1.5, 2.5], [3.5, 4.5]]);
$debug = $grid->__debugInfo();
check($debug === ['kind' => 'tensor', 'type' => 'float32', 'elements' => 4, 'shape' => [2, 2]], 'debug info is the whole tensor');
check(json_encode($grid) === '[[1.5,2.5],[3.5,4.5]]', 'json nests by shape');
check(count($grid) === 4, 'count is the element count');
check(str_contains(message(throws(fn () => $grid->getIterator())), 'sequence or a map'), 'tensor is not iterable');
check(str_contains(message(throws(fn () => json_encode(OnnxValue::fromBytes(pack('f', 1.0))))), 'fromBytes'), 'packed value has no json');
check(count(OnnxValue::tensor(['a', 'b'])) === 2, 'count counts strings');

$roundTrip = unserialize(serialize($grid));
check($roundTrip->values() === [1.5, 2.5, 3.5, 4.5] && $roundTrip->shape() === [2, 2] && $roundTrip->elementType() === 'float32', 'tensor survives a cache round trip');
$words = OnnxValue::tensor(['cat', 'dog']);
check(unserialize(serialize($words))->values() === ['cat', 'dog'], 'string tensor survives a round trip');
$packed = OnnxValue::fromBytes(pack('f2', 1.0, 2.0));
check(unserialize(serialize($packed))->toBytes() === $packed->toBytes(), 'packed bytes survive a round trip');

$walk = (new OnnxSession($dir . '/seq.onnx'))->run(['x' => [1.0, 2.0, 3.0]])['seq'];
check(json_encode($walk) === '[[[1,2,3]]]', 'sequence json');
$seen = 0;
foreach ($walk as $item) {
    $seen += $item instanceof OnnxValue ? 1 : 0;
}
check($seen === 1 && count($walk) === 1, 'sequence iterates and counts');
$labels = (new OnnxSession($dir . '/zipmap.onnx'))->run(['x' => [[0.25, 0.75]]])['scores']->items()[0];
check(json_encode($labels) === '{"no":[0.25],"yes":[0.75]}', 'map json is keyed');
$pairs = 0;
foreach ($labels as $pair) {
    $pairs += $pair[1] instanceof OnnxValue ? 1 : 0;
}
check($pairs === 2 && count($labels) === 2, 'map iterates pairs');
check(str_contains(message(throws(fn () => serialize($walk))), 'supports a tensor'), 'sequence does not serialize');

check(OnnxValue::tensor([1, 2], 'int16')->elementType() === 'int16', 'int16 tensor');
check(OnnxValue::tensor([1, 2], 'uint16')->elementType() === 'uint16', 'uint16 tensor');
check(OnnxValue::tensor([1, 2], 'uint32')->elementType() === 'uint32', 'uint32 tensor');
check(OnnxValue::tensor([1, 2], 'float64')->elementType() === 'float64', 'float64 tensor');
check((new OnnxSession($dir . '/i16.onnx'))->run(['x' => [1, 2]])['y']->values() === [1, 2], 'int16 model feed');
check((new OnnxSession($dir . '/u16.onnx'))->run(['x' => [1, 2]])['y']->values() === [1, 2], 'uint16 model feed');
check((new OnnxSession($dir . '/u32.onnx'))->run(['x' => [1, 2]])['y']->values() === [1, 2], 'uint32 model feed');
check(str_contains(message(throws(fn () => OnnxValue::tensor([40000], 'int16'))), 'range'), 'int16 range is checked');
check(str_contains(message(throws(fn () => (new OnnxSession($dir . '/bf16.onnx'))->run(['x' => [1.0, 2.0]]))), 'bfloat16 must be packed'), 'bfloat16 names itself');
check(str_contains(message(throws(fn () => OnnxValue::tensor([1], 'uint64'))), 'fromBytes'), 'unknown type points at fromBytes');

$six = OnnxValue::fromBytes(pack('f6', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
$twoFree = new OnnxSession($dir . '/twodyn.onnx');
check(str_contains(message(throws(fn () => $twoFree->run(['x' => $six]))), 'two or more dynamic dimensions'), 'two free dimensions need a name');
$oneRow = new OnnxSession($dir . '/twodyn.onnx', (new OnnxOptions())->freeDimensions(['batch' => 1]));
check($oneRow->run(['x' => $six])['y']->shape() === [1, 6], 'free dimension fixes the row');
$twoRows = new OnnxSession($dir . '/twodyn.onnx', (new OnnxOptions())->freeDimensions(['batch' => 2]));
check($twoRows->run(['x' => $six])['y']->shape() === [2, 3], 'free dimension sets the row count');
$named = new OnnxSession($identity, (new OnnxOptions())->freeDimensions(['nope' => 3]));
check($named->inputs()[0]->shape() === ['batch', 3], 'unknown dimension name is ignored');
check(str_contains(message(throws(fn () => (new OnnxOptions())->freeDimensions(['batch' => 0]))), 'positive'), 'free dimension size is positive');
check(str_contains(message(throws(fn () => (new OnnxOptions())->freeDimensions([]))), 'must not be empty'), 'free dimensions need an entry');
check((new OnnxOptions())->freeDimensions(['batch' => 1])->intraOpThreads(2) instanceof OnnxOptions, 'free dimensions chain');

$closing = new OnnxSession($identity);
$kept = $closing->run(['x' => [1.0, 2.0, 3.0]])['y'];
$closing->close();
check(str_contains(message(throws(fn () => $closing->run(['x' => [1.0, 2.0, 3.0]]))), 'after close'), 'run after close');
check(str_contains(message(throws(fn () => $closing->inputs())), 'after close'), 'inputs after close');
check(str_contains(message(throws(fn () => $closing->metadata())), 'after close'), 'metadata after close');
check($closing->providers() === ['cpu'] && $kept->values() === [1.0, 2.0, 3.0], 'close keeps providers and values');
$closing->close();
check(true, 'close is idempotent');

if ($failures > 0) {
    fwrite(STDERR, "{$failures} failed\n");
    exit(1);
}
echo "ok\n";
