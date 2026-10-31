# Switon\OnnxRuntime

IDE stubs. One class per file. The directory matches the namespace. Do not `include` these files and do not Composer-autoload them. The extension provides the same classes at runtime. They need PHP 8.2 through 8.5.

## Making the IDE see them

There is no install step. A compiled extension has no source for an editor to read, so these files are the signature: the editor indexes the directory, and PHP never includes it. Point the editor's include path at this folder, and completion, parameter hints, and return types all work.

Keep the folder out of `composer.json` autoload. A stub method has no body, so if the autoloader ever did load one, which happens only when the extension is missing, the call would fail as a `TypeError` from an empty stub instead of the honest `class not found`.

**PhpStorm.** Settings → PHP → Include Path → `+` → the `stubs` directory. The setting is stored in `.idea`, so it is per project. Directories under `vendor/` are indexed already.

**VS Code with Intelephense.** In the project's `.vscode/settings.json`, with the path to wherever this repository sits:

```json
{
  "intelephense.environment.includePaths": ["/path/to/ext-onnxruntime/stubs"]
}
```

**Another editor.** The setting is usually called include path, extra path, or stub directory. Point it at the same folder.

To confirm it took, type `OnnxSession::` in an empty file: the list should offer `run()`, `inputs()`, and `fromContents()`. Without the include path the name is reported as undefined.

The surface is ONNX Runtime's session: load a model, inspect inputs and outputs, run it. After `tensor()` or `fromBytes()`, the value's bytes stay in the extension. `tensor()` copies a PHP array in. `values()` copies elements back out. `fromBytes()` and `toBytes()` do not build one PHP value per element. Thread counts, graph optimization, parallel execution, log level, and the provider list are the knobs everyday code sets. `OnnxValue` constants are the element type names. `tensor()` takes one of them, except `FLOAT16`, which is packed bytes. `elementType()` may still return another string, such as `float64`. Allocators, IOBinding, custom operators, sparse tensors, and profiling are not classes here. Reuse an `OnnxValue` across `OnnxSession::run()` calls instead of rebinding buffers.

A session with no options uses CPU only. A provider reported by `Build::availableProviders()` can still fail to load. Pass CUDA only on a machine that has it. `OnnxSession` copies `OnnxOptions` at construction. `run()` blocks the calling PHP thread. Another OS thread may call it on the same session. Do not change a feed until that call returns. A value from `run()` keeps its own bytes after the session is gone. `close()` releases the graph and its arena at that line, for a worker that swaps models; `run()`, `inputs()`, `outputs()`, and `metadata()` throw afterwards.

## Recommended use

```php
$options = (new OnnxOptions())
    ->providers(['cuda', 'cpu'])
    ->intraOpThreads(4);

$session = new OnnxSession('/models/encoder.onnx', $options);
$hidden = $session->run([
    'input_ids' => $ids,
    'attention_mask' => 1,
], ['last_hidden_state'])['last_hidden_state'];
$shape = $hidden->shape();
$bytes = $hidden->toBytes();

$next = $decoder->run([
    'input_ids' => $generated,
    'encoder_attention_mask' => 1,
    'encoder_hidden_states' => $hidden,
], [
    'logits' => ['index' => 'max'],
])['logits']->value();
```

`1` fills every mask position. The shape follows the other feeds that share its dimension names. `$decoder` is a second session. `$generated` is the token ids so far, and the caller appends `$next` and calls `run()` again. `value()` is that one id when the batch has one row. Several rows come back from `values()`. `['index' => 'max', 'step' => 'each']` returns one index per step. `'step'` without `'index'` throws.

`$ids` is a list of ints. The model input declares the rank, so a list bound to `[batch, sequence]` is one batch row. A list of lists is an explicit batch. A packed image is a string. The input's element type decides how many bytes each element takes. The shape decides how many elements there are. The byte length has to match. NCHW and NHWC can have the same length. The string has to already be in the model's layout. A string tensor takes a list of strings. A PHP string in the feed is packed bytes, not one string element.

A PHP array or string is always a tensor. Token ids are small enough to pass as a list. An image or a hidden state stays a packed string: `fromBytes()` in, `toBytes()` out. `values()` copies every element into one flat list, row-major. Use it for a short vector, such as class scores. A sequence or a map comes back from `run()` as an `OnnxValue`. Pass that same value when a later input expects one. `items()` and `entries()` return those values. They do not copy each tensor element into PHP. `null` and `[]` both fetch every output. Naming an output chooses which values come back. The session still runs the graph.

A sequence and a map nest one level. ONNX Runtime registers no type for a sequence of sequences, so a model declaring `sequence<sequence<float32>>` fails to load before this extension sees it. That is not a limit this extension chose.

A value also speaks the PHP data protocols. `json_encode()` returns the nesting `shape()` describes, so a `[2, 2]` tensor is two lists of two values. `count()` is the element count, and a sequence or a map is iterable with `foreach`. `var_dump()` shows kind, type, and shape rather than an empty object. `serialize()` keeps the element type, the shape, and the bytes, so a tensor can go into a cache and come back as the same tensor. A sequence and a map are not serialized: walk them first. A tensor is not iterable; `foreach` on one throws instead of looping zero times.

A path load sees external weight files next to the `.onnx`. `OnnxSession::fromContents()` does not.

## Moving from Python or ankane/onnxruntime

`providers()` takes a short name from `availableProviders()` or `OnnxSession::providers()`, and the same id ending in `ExecutionProvider`. `cpu`, `cuda`, `coreml`, and `tensorrt` are accepted even when this build does not list them. Option maps are passed through as strings.

`ankane/onnxruntime` runs a model in one call: `$model->predict(['x' => $row])`. Here a session is built once and `run()` is called for each input, so the graph is compiled once.

`run()` takes the feeds first and the output names second. Python takes the output names first. Both arguments are arrays, so a swapped call fails as an unknown name.

`run()` returns a map of output name to `OnnxValue`. Python returns a list. Read `$result['last_hidden_state']`, not `$result[0]`.

`OnnxValue::elementType()` is `float32` or `int64`, not Python's `tensor(float)`. `OnnxValueInfo::type()` is the declaration: a tensor is `float32`, a sequence or a map is `sequence<float32>` or `map<int64,float32>`. The container alone is `kind()`.

`values()` is one flat list. One element is `value()`. `values()[0]` is the first scalar, not the first row. Keep `shape()` beside it. A 1000-way logit vector can be used as that flat list.

A PHP string in a feed is packed bytes. A string element is a one-element list, or `OnnxValue::tensor(['hello'])`.

`tensor([0, 1, 2])` is int64. A uint8 image needs `OnnxValue::UINT8`, or pass the bare list to `run()` and let the model input's type convert it. `float64`, `int16`, `uint16`, and `uint32` are accepted by name. `FLOAT16`, `bfloat16`, and `uint64` are packed bytes via `OnnxValue::fromBytes()`, not `tensor()`: PHP has no 16-bit float, and a PHP int cannot hold the top half of a uint64.

A list one rank below a dynamic batch becomes one row. A model that fixed that dimension at export does not accept the shorter list.

A classifier sequence of maps is not a PHP list of maps. Walk `items()`, then `entries()`. A score of one element is `value()`. `items()` throws on a tensor such as logits.

`new OnnxSession($path)` uses CPU only. Python's `InferenceSession(path)` may already include CUDA. Pass `providers(['cuda', 'cpu'])` to use it.

`OnnxSession::fromContents()` loads a model file. `OnnxValue::fromBytes()` is packed tensor bytes. The two names differ because the two arguments mean different things: a whole model, or one input buffer. A path load sees weight files next to the `.onnx`. The model-content load does not.

## Scope

One forward pass: load, inspect, feed, run, read. Tokenizing and image decoding stay in PHP. An attention mask of ones, a sequence mask, and the index of the largest logit are requests on `run()`. The caller writes the loop that feeds the next token id back in. A KV cache is a tensor the caller passes on the next `run()`. Two or more free dimensions cannot be fed from packed bytes unless `OnnxOptions::freeDimensions()` names a size for all but one of them: otherwise use a PHP array, or a model with one free dimension.

## Classes

| Class | Role |
| --- | --- |
| `Switon\OnnxRuntime\Build` | Linked ONNX Runtime version and the providers this build has. |
| `Switon\OnnxRuntime\OnnxSession` | Loaded model. `run()` accepts an `OnnxValue`, a PHP array, a packed string, or a mask request such as `1`. An output map such as `['logits' => ['index' => 'max']]` returns the winning index. One id is `value()`. `close()` frees the model before the object goes away. |
| `Switon\OnnxRuntime\OnnxOptions` | Thread counts, graph optimization, parallel execution, log level, provider list, free dimension sizes. Copied when the session is constructed. |
| `Switon\OnnxRuntime\OptimizationLevel` | `Off`, `Basic`, `Extended`, `All`. |
| `Switon\OnnxRuntime\LogLevel` | `Verbose`, `Info`, `Warning`, `Error`, `Fatal`. |
| `Switon\OnnxRuntime\OnnxValue` | Tensor, sequence, or map held by the extension. |
| `Switon\OnnxRuntime\OnnxValueInfo` | Name, kind, declaration, and shape of one input or output. A named dynamic dim is that name, such as `batch`. An unnamed one is `null`. |
| `Switon\OnnxRuntime\OnnxMetadata` | Producer, graph name, domain, description, version, and the custom metadata map. |
| `Switon\OnnxRuntime\OnnxException` | Thrown by `OnnxSession` and `OnnxValue`. `OnnxOptions` only records settings. |
