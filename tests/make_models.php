<?php

declare(strict_types=1);

/**
 * Write the tiny ONNX models used by tests and examples.
 * Plain PHP. The extension is not required.
 */

$root = dirname(__DIR__);
$out = $root . '/tests/models';
$examples = $root . '/examples/models';

const FLOAT = 1;
const INT64 = 7;
const STRING = 8;
const FLOAT16 = 10;

function varint(int $number): string
{
    $out = '';
    while (true) {
        $byte = $number & 0x7F;
        $number >>= 7;
        if ($number !== 0) {
            $out .= chr($byte | 0x80);
        } else {
            $out .= chr($byte);
            return $out;
        }
    }
}

function field(int $number, int $wire, string $payload): string
{
    return varint(($number << 3) | $wire) . $payload;
}

function proto_int(int $number, int $value): string
{
    return field($number, 0, varint($value));
}

function proto_bytes(int $number, string $payload): string
{
    return field($number, 2, varint(strlen($payload)) . $payload);
}

function proto_str(int $number, string $text): string
{
    return proto_bytes($number, $text);
}

function proto_msg(int $number, string $body): string
{
    return proto_bytes($number, $body);
}

function dimension(int|string|null $value): string
{
    if (is_int($value)) {
        $body = proto_int(1, $value);
    } elseif (is_string($value)) {
        $body = proto_str(2, $value);
    } else {
        $body = '';
    }
    return proto_msg(1, $body);
}

/** @param list<int|string|null> $dims */
function tensor_type(int $elem, array $dims): string
{
    $shape = '';
    foreach ($dims as $dim) {
        $shape .= dimension($dim);
    }
    $tensor = proto_int(1, $elem) . proto_msg(2, $shape);
    return proto_msg(1, $tensor);
}

/** @param list<int|string|null> $dims */
function value_info(string $name, int $elem, array $dims): string
{
    return proto_str(1, $name) . proto_msg(2, tensor_type($elem, $dims));
}

/** @param list<string> $inputs @param list<string> $outputs */
function node(string $op, array $inputs, array $outputs, string $name = 'n', ?string $domain = null, string $attributes = ''): string
{
    $body = '';
    foreach ($inputs as $item) {
        $body .= proto_str(1, $item);
    }
    foreach ($outputs as $item) {
        $body .= proto_str(2, $item);
    }
    $body .= proto_str(3, $name) . proto_str(4, $op) . $attributes;
    if ($domain !== null) {
        $body .= proto_str(7, $domain);
    }
    return $body;
}

/** @param list<string> $nodes @param list<string> $inputs @param list<string> $outputs */
function graph(string $name, array $nodes, array $inputs, array $outputs, string $initializers = ''): string
{
    $body = '';
    foreach ($nodes as $item) {
        $body .= proto_msg(1, $item);
    }
    $body .= proto_str(2, $name) . $initializers;
    foreach ($inputs as $item) {
        $body .= proto_msg(11, $item);
    }
    foreach ($outputs as $item) {
        $body .= proto_msg(12, $item);
    }
    return $body;
}

/** @param array<string, string>|null $metadata @param list<array{0: string, 1: int}>|null $opsets */
function model(string $graphBody, string $producer = 'test', string $domain = 'test.domain', int $version = 3, string $doc = 'identity', ?array $metadata = null, ?array $opsets = null): string
{
    $body = proto_int(1, 8);
    $body .= proto_str(2, $producer);
    $body .= proto_str(4, $domain);
    $body .= proto_int(5, $version);
    $body .= proto_str(6, $doc);
    $body .= proto_msg(7, $graphBody);
    foreach ($opsets ?? [['', 13]] as [$opDomain, $opVersion]) {
        $opset = proto_str(1, $opDomain) . proto_int(2, $opVersion);
        $body .= proto_msg(8, $opset);
    }
    foreach ($metadata ?? [] as $key => $value) {
        $entry = proto_str(1, $key) . proto_str(2, $value);
        $body .= proto_msg(14, $entry);
    }
    return $body;
}

function save(string $dir, string $name, string $payload): void
{
    if (!is_dir($dir) && !mkdir($dir, 0777, true) && !is_dir($dir)) {
        fwrite(STDERR, "cannot create {$dir}\n");
        exit(1);
    }
    file_put_contents($dir . '/' . $name, $payload);
}

/** @param list<float> $values */
function raw_floats(array $values): string
{
    return pack('g*', ...$values);
}

function raw_i64(int $value): string
{
    $lo = $value & 0xFFFFFFFF;
    $hi = ($value >> 32) & 0xFFFFFFFF;
    return pack('V2', $lo, $hi);
}

/** @param list<int> $dims */
function tensor_proto(string $name, array $dims, string $raw, int $elem = 1): string
{
    $body = '';
    foreach ($dims as $dim) {
        $body .= proto_int(1, $dim);
    }
    $body .= proto_int(2, $elem);
    $body .= proto_str(8, $name);
    $body .= proto_bytes(9, $raw);
    return proto_msg(5, $body);
}

function attr_int(string $name, int $value): string
{
    $body = proto_str(1, $name) . proto_int(20, 2) . proto_int(3, $value);
    return proto_msg(5, $body);
}

/** @param list<int> $values */
function attr_ints(string $name, array $values): string
{
    $body = proto_str(1, $name) . proto_int(20, 7);
    foreach ($values as $value) {
        $body .= proto_int(8, $value);
    }
    return proto_msg(5, $body);
}

function write_tests(string $dir): void
{
    save($dir, 'identity.onnx', model(graph(
        'g',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT, ['batch', 3])],
        [value_info('y', FLOAT, ['batch', 3])],
    ), metadata: ['labels' => 'cat,dog']));
    save($dir, 'fixed.onnx', model(graph(
        'fixed',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT, [2, 3])],
        [value_info('y', FLOAT, [2, 3])],
    ), doc: 'fixed'));
    save($dir, 'unnamed.onnx', model(graph(
        'unnamed',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT, [null, 4])],
        [value_info('y', FLOAT, [null, 4])],
    ), doc: 'unnamed'));
    save($dir, 'strings.onnx', model(graph(
        'strings',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', STRING, ['n'])],
        [value_info('y', STRING, ['n'])],
    ), doc: 'strings'));
    save($dir, 'ints.onnx', model(graph(
        'ints',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', INT64, ['n'])],
        [value_info('y', INT64, ['n'])],
    ), doc: 'ints'));
    save($dir, 'two.onnx', model(graph(
        'two',
        [node('Identity', ['x'], ['y'], 'ny'), node('Identity', ['x'], ['z'], 'nz')],
        [value_info('x', FLOAT, ['batch', 3])],
        [value_info('y', FLOAT, ['batch', 3]), value_info('z', FLOAT, ['batch', 3])],
    ), doc: 'two'));
    save($dir, 'add.onnx', model(graph(
        'add',
        [node('Add', ['x', 'b'], ['y'])],
        [value_info('x', FLOAT, ['batch', 3]), value_info('b', FLOAT, [3])],
        [value_info('y', FLOAT, ['batch', 3])],
        tensor_proto('b', [3], raw_floats([1.0, 1.0, 1.0])),
    ), doc: 'add'));
    $sequence = proto_msg(4, proto_msg(1, tensor_type(FLOAT, ['batch', 3])));
    $seqInfo = proto_str(1, 'seq') . proto_msg(2, $sequence);
    save($dir, 'seq.onnx', model(graph(
        'seq',
        [node('SequenceConstruct', ['x'], ['seq'])],
        [value_info('x', FLOAT, ['batch', 3])],
        [$seqInfo],
    ), doc: 'seq'));
    // A sequence whose elements have different shapes. The element type carries no
    // shape, which is the whole point of a sequence: ragged rows.
    // TypeProto{sequence{Sequence{elem_type:TypeProto{tensor_type{elem_type}}}}}
    $raggedSeq = proto_msg(4, proto_msg(1, proto_msg(1, proto_int(1, FLOAT))));
    $raggedInfo = proto_str(1, 'ragged') . proto_msg(2, $raggedSeq);
    save($dir, 'ragged.onnx', model(graph(
        'ragged',
        [node('SequenceConstruct', ['a', 'b', 'c'], ['ragged'])],
        [],
        [$raggedInfo],
        tensor_proto('a', [1], raw_floats([1.0]))
            . tensor_proto('b', [2], raw_floats([2.0, 3.0]))
            . tensor_proto('c', [3], raw_floats([4.0, 5.0, 6.0])),
    ), doc: 'ragged'));
    $stringsAttr = proto_str(1, 'classlabels_strings') . proto_int(20, 8);
    $stringsAttr .= proto_bytes(9, 'no') . proto_bytes(9, 'yes');
    $mapValue = tensor_type(FLOAT, []);
    $mapType = proto_int(1, STRING) . proto_msg(2, $mapValue);
    $sequenceOfMaps = proto_msg(4, proto_msg(1, proto_msg(5, $mapType)));
    $zipOut = proto_str(1, 'scores') . proto_msg(2, $sequenceOfMaps);
    save($dir, 'f16.onnx', model(graph(
        'f16',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT16, [2])],
        [value_info('y', FLOAT16, [2])],
    ), doc: 'f16'));
    save($dir, 'twodyn.onnx', model(graph(
        'twodyn',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT, ['batch', 'seq'])],
        [value_info('y', FLOAT, ['batch', 'seq'])],
    ), doc: 'twodyn'));
    $position = proto_int(2, INT64) . proto_str(8, 'pos') . proto_bytes(9, raw_i64(0));
    save($dir, 'seq_at.onnx', model(graph(
        'seq_at',
        [node('SequenceAt', ['seq', 'pos'], ['y'])],
        [$seqInfo],
        [value_info('y', FLOAT, ['batch', 3])],
        proto_msg(5, $position),
    ), doc: 'seq_at'));
    foreach ([
        'i8' => 3, 'u8' => 2, 'i16' => 5, 'u16' => 4, 'i32' => 6,
        'u32' => 12, 'u64' => 13, 'f64' => 11, 'bools' => 9, 'bf16' => 16,
    ] as $name => $elem) {
        save($dir, $name . '.onnx', model(graph(
            $name,
            [node('Identity', ['x'], ['y'])],
            [value_info('x', $elem, [2])],
            [value_info('y', $elem, [2])],
        ), doc: $name));
    }
    save($dir, 'onedyn.onnx', model(graph(
        'onedyn',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT, [null])],
        [value_info('y', FLOAT, [null])],
    ), doc: 'onedyn'));
    $optional = proto_msg(9, proto_msg(1, tensor_type(FLOAT, [3])));
    save($dir, 'optional.onnx', model(graph(
        'optional',
        [node('OptionalHasElement', ['x'], ['y'])],
        [proto_str(1, 'x') . proto_msg(2, $optional)],
        [value_info('y', 9, [])],
    ), doc: 'optional', opsets: [['', 18]]));
    save($dir, 'gather.onnx', model(graph(
        'gather',
        [node('Gather', ['x', 'i'], ['y'])],
        [value_info('x', FLOAT, [3]), value_info('i', INT64, [1])],
        [value_info('y', FLOAT, [1])],
    ), doc: 'gather'));
    foreach (['f8' => 17, 'f8fnuz' => 18, 'f8e5' => 19, 'f8e5fnuz' => 20, 'i4' => 22] as $name => $elem) {
        $castTo = proto_str(1, 'to') . proto_int(20, 2) . proto_int(3, $elem);
        save($dir, $name . '.onnx', model(graph(
            $name,
            [node('Cast', ['x'], ['y'], attributes: proto_msg(5, $castTo))],
            [value_info('x', FLOAT, [2])],
            [value_info('y', $elem, [2])],
        ), doc: $name));
    }
    $bare = proto_int(1, 8) . proto_msg(7, graph(
        'bare',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT, [1])],
        [value_info('y', FLOAT, [1])],
    ));
    $bare .= proto_msg(8, proto_int(2, 13));
    save($dir, 'bare.onnx', $bare);
    $dup = model(graph(
        'dup',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT, [1])],
        [value_info('y', FLOAT, [1])],
    ), doc: 'dup', metadata: ['labels' => 'second']);
    $from = proto_msg(14, proto_str(1, 'labels') . proto_str(2, 'second'));
    $to = proto_msg(14, proto_str(1, 'labels') . proto_str(2, 'first'))
        . proto_msg(14, proto_str(1, 'labels') . proto_str(2, 'second'));
    save($dir, 'dup.onnx', str_replace($from, $to, $dup));
    $intLabels = proto_str(1, 'classlabels_int64s') . proto_int(20, 7);
    $intLabels .= proto_int(8, 10) . proto_int(8, 20);
    $intMap = proto_int(1, INT64) . proto_msg(2, tensor_type(FLOAT, []));
    $intZip = proto_str(1, 'scores') . proto_msg(2, proto_msg(4, proto_msg(1, proto_msg(5, $intMap))));
    save($dir, 'zipint.onnx', model(graph(
        'zipint',
        [node('ZipMap', ['x'], ['scores'], domain: 'ai.onnx.ml', attributes: proto_msg(5, $intLabels))],
        [value_info('x', FLOAT, [1, 2])],
        [$intZip],
    ), doc: 'zipint', opsets: [['', 13], ['ai.onnx.ml', 1]]));
    $castMap = proto_int(1, INT64) . proto_msg(2, tensor_type(FLOAT, ['n']));
    $castIn = proto_str(1, 'x') . proto_msg(2, proto_msg(5, $castMap));
    $castAttrs = proto_msg(5, proto_str(1, 'cast_to') . proto_int(20, 3) . proto_bytes(4, 'TO_FLOAT'));
    $castAttrs .= proto_msg(5, proto_str(1, 'map_form') . proto_int(20, 3) . proto_bytes(4, 'DENSE'));
    $castAttrs .= proto_msg(5, proto_str(1, 'max_map') . proto_int(20, 2) . proto_int(3, 2));
    save($dir, 'castmap.onnx', model(graph(
        'castmap',
        [node('CastMap', ['x'], ['y'], domain: 'ai.onnx.ml', attributes: $castAttrs)],
        [$castIn],
        [value_info('y', FLOAT, [1, 2])],
    ), doc: 'castmap', opsets: [['', 13], ['ai.onnx.ml', 1]]));
    save($dir, 'huge.onnx', model(graph(
        'huge',
        [node('Identity', ['x'], ['y'])],
        [value_info('x', FLOAT, [2 ** 62, 8])],
        [value_info('y', FLOAT, [2 ** 62, 8])],
    ), doc: 'huge'));
    save($dir, 'zipmap.onnx', model(graph(
        'zip',
        [node('ZipMap', ['x'], ['scores'], domain: 'ai.onnx.ml', attributes: proto_msg(5, $stringsAttr))],
        [value_info('x', FLOAT, [1, 2])],
        [$zipOut],
    ), doc: 'zipmap', opsets: [['', 13], ['ai.onnx.ml', 1]]));
    save($dir, 'maskio.onnx', model(graph(
        'maskio',
        [node('Identity', ['encoder_attention_mask'], ['mask'])],
        [
            value_info('input_ids', INT64, ['batch', 'seq']),
            value_info('encoder_attention_mask', INT64, ['batch', 'seq']),
        ],
        [value_info('mask', INT64, ['batch', 'seq'])],
    ), doc: 'mask'));
    save($dir, 'logits.onnx', model(graph(
        'logits',
        [node('Identity', ['scores'], ['logits'])],
        [value_info('scores', FLOAT, ['batch', 'step', 'vocab'])],
        [value_info('logits', FLOAT, ['batch', 'step', 'vocab'])],
    ), doc: 'logits'));
}

function write_examples(string $dir): void
{
    $hidden = 4;
    $table = [];
    for ($token = 0; $token < 128; $token++) {
        array_push($table, $token / 127.0, ($token % 5) / 4.0, $token !== 0 ? 1.0 : 0.0, 0.25);
    }
    $axes = tensor_proto('mask_axes', [1], raw_i64(2), INT64);
    save($dir, 'encoder.onnx', model(graph(
        'encoder',
        [
            node('Gather', ['table', 'input_ids'], ['gathered'], 'gather'),
            node('Cast', ['attention_mask'], ['mask_f'], 'cast', attributes: attr_int('to', FLOAT)),
            node('Unsqueeze', ['mask_f', 'mask_axes'], ['mask_u'], 'unsqueeze'),
            node('Mul', ['gathered', 'mask_u'], ['last_hidden_state'], 'mul'),
        ],
        [value_info('input_ids', INT64, ['batch', 'seq']), value_info('attention_mask', INT64, ['batch', 'seq'])],
        [value_info('last_hidden_state', FLOAT, ['batch', 'seq', $hidden])],
        tensor_proto('table', [128, $hidden], raw_floats($table)) . $axes,
    ), metadata: ['hidden' => (string) $hidden], doc: 'token embedding lookup'));
    save($dir, 'classifier.onnx', model(graph(
        'classifier',
        [
            node('MatMul', ['input', 'W'], ['mm'], 'matmul'),
            node('Add', ['mm', 'b'], ['logits'], 'add'),
        ],
        [value_info('input', FLOAT, ['batch', $hidden])],
        [value_info('logits', FLOAT, ['batch', 3])],
        tensor_proto('W', [$hidden, 3], raw_floats([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.2, 0.2, 0.2,
        ])) . tensor_proto('b', [3], raw_floats([0.0, 0.1, -0.1])),
    ), metadata: ['labels' => 'cat,dog,bird'], doc: 'linear classifier'));
    $pixels = [];
    foreach ([0.1, 0.2, 0.9] as $level) {
        $pixels = array_merge($pixels, array_fill(0, 8 * 8, $level));
    }
    save($dir, 'input.f32', raw_floats($pixels));
    save($dir, 'resnet.onnx', model(graph(
        'resnet',
        [
            node('ReduceMean', ['input'], ['pooled'], 'pool', attributes: attr_ints('axes', [2, 3]) . attr_int('keepdims', 0)),
            node('MatMul', ['pooled', 'W'], ['logits'], 'matmul'),
        ],
        [value_info('input', FLOAT, ['batch', 3, 8, 8])],
        [value_info('logits', FLOAT, ['batch', 4])],
        tensor_proto('W', [3, 4], raw_floats([
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ])),
    ), metadata: ['labels' => 'cat,dog,bird,other'], doc: 'nchw average pool'));
}

write_tests($out);
write_examples($examples);
