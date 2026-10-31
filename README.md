# ext-onnxruntime

PHP extension. `Switon\OnnxRuntime` runs an ONNX model with ONNX Runtime 1.31.1. Supported platforms are macOS Apple Silicon, Linux glibc (`x86_64` and `arm64`), and Windows x64, PHP 8.2–8.5, NTS and ZTS. Contributor conventions are in [AGENTS.md](AGENTS.md).

## When to use this, and when not

It puts the model in the PHP process, which removes the HTTP hop to a Python service. It removes the hop, not the server.

**Reach for it when** the model is small and the call is on the request path:

- a classifier, an embedding model, or a reranker that answers in single-digit milliseconds
- a queue worker or a batch job that already runs PHP and would otherwise shell out
- a service that cannot take a second runtime, a second deploy, or a second thing to page on-call for

**Do not reach for it when:**

- **the model is large.** [bench/](bench/README.md) measures a BERT-base-sized graph at 6.3 ms in-process and 6.7 ms over loopback HTTP. The boundary stops mattering, and a Python service gives you batching, GPU, and an ecosystem you would have to rebuild.
- **you need tokenizing, sampling, or a generation loop.** None of that is here. Tokenizers and image decoding stay in PHP, and there is no sampling, no KV-cache management, and no detokenizing. This runs a graph; it is not `transformers`.
- **you need CUDA.** The published packages are the official default builds, not the CUDA builds. `cuda` works only when the library you link is a CUDA build. The macOS package includes CoreML and WebGPU. `Build::availableProviders()` lists the short names this process can register, and those names go into `providers()`.
- **the deployment cannot compile C.** Until a `v*` tag exists there is no prebuilt binary, so `pie install` compiles. See [Build](#build).

## Install

Build from this repository. The steps are in [Build](#build). A `v*` tag publishes a prebuilt file; `pie install switon/onnxruntime` uses that file when the PHP version matches, and compiles otherwise. Until a tag exists, there is no file to download. A source build needs the ONNX Runtime C library below. The extension links that shared library. It does not carry its own copy of the runtime inside the `.so`.

The classes are provided by the extension. [`stubs/`](stubs/README.md) is the same surface for an editor, and the per-editor steps to make it visible are in that file. Nothing installs it: add the directory to the IDE include path, and do not `include` it or add it to Composer autoload.

## Quick start

This is not a one-to-one binding of the ONNX Runtime C API or of the Python `InferenceSession`. There is no `OrtValue` constructor, no name array, and no numpy dtype on the call. Open one session, pass PHP values, and read `OnnxValue` results. Tokenizing and image decoding stay in PHP. The method list is [stubs/README.md](stubs/README.md).

Point it at a model you already have. The session reads the input names and shapes out of the file, so the first call is a dump rather than a guess:

```php
use Switon\OnnxRuntime\OnnxSession;

$session = new OnnxSession('/path/to/model.onnx');

foreach ($session->inputs() as $input) {
    echo $input->name(), ' ', $input->type(), ' ', json_encode($input->shape()), "\n";
}

$result = $session->run(['x' => [1.0, 2.0, 3.0]]);   // the key is the name printed above
echo json_encode($result[array_key_first($result)]->values()), "\n";
```

With a real input and a thread count:

```php
use Switon\OnnxRuntime\OnnxOptions;
use Switon\OnnxRuntime\OnnxSession;

$options = (new OnnxOptions())
    ->providers(['cpu'])
    ->intraOpThreads(4);

$session = new OnnxSession('/path/to/model.onnx', $options);
$hidden = $session->run([
    'input_ids' => $ids,
    'attention_mask' => 1,
], ['last_hidden_state'])['last_hidden_state'];
```

No model of your own to try? `make examples` writes a small encoder to `examples/models/`, then runs every script in `examples/`:

```bash
make examples
php -d extension="$PWD/pie/modules/onnxruntime.so" examples/inspect.php
```

`inspect.php` prints the inputs. `embed.php` is the same session with a real batch of token ids. What each script does is in [examples/README.md](examples/README.md). `examples/models/encoder.onnx` is written by `make examples`, not stored in git.

Keep that session and call `run()` again. Creating it compiles the graph. A list of ints is one batch row when the leading dimension is free. The integer `1` fills an input whose name ends in `attention_mask`. The result is a map of output name to `OnnxValue`, not a list: read `$hidden`, not `$result[0]`. A short vector is `values()`. One number is `value()`. A hidden state or an image stays `toBytes()`.

The next token id is a request on the output, not a PHP loop over the logits:

```php
$next = $decoder->run([
    'input_ids' => $generated,
    'encoder_attention_mask' => 1,
    'encoder_hidden_states' => $hidden,
], [
    'logits' => ['index' => 'max'],
])['logits']->value();
```

`$decoder` is a second session. Append `$next` to `$generated` and call `run()` again. `['index' => 'max', 'step' => 'each']` returns one index per step. `'step'` without `'index'` throws.

| Python or the C API | Here |
| --- | --- |
| `InferenceSession` plus a numpy array, dtype, and shape | `OnnxSession` plus a PHP list or a packed string. The model input supplies the element type and the rank. |
| `np.ones` for a mask | `attention_mask => 1` |
| `outputs[0]` | `$result['last_hidden_state']` |
| `logits.argmax()` | `['logits' => ['index' => 'max']]`, then `value()` |
| Tokenizer, image decoder, the decode loop | PHP. The extension does not own them. |

## Build

The library matches the `php` and `php-config` on `PATH` while it compiles. Those two commands must print the same version. On Windows there is no `php-config`; the build reads the `php` on `PATH`. The result loads only on that PHP minor version, that NTS or ZTS build, and that CPU. The ONNX Runtime prefix is the official 1.31.1 release. `ORT` points at the directory that contains `include/` and `lib/`.

From a clone of this repository, with `ORT` set:

```bash
make test
```

On macOS Apple Silicon, an unset `ORT` uses `deps/onnxruntime-osx-arm64-1.31.1`. That directory is not in git.

`make test` builds the extension and loads it. The file is `pie/modules/onnxruntime.so` on macOS and Linux, and `php_onnxruntime.dll` on Windows. `php -m` lists the module as `onnxruntime`. The classes stay under `Switon\OnnxRuntime`.

```bash
php -d extension="$PWD/pie/modules/onnxruntime.so" -m
make examples
php -d extension="$PWD/pie/modules/onnxruntime.so" examples/inspect.php
```

`make examples` writes the small graphs under `examples/models/` and runs every script in `examples/`. `inspect.php` prints the linked ONNX Runtime version and the encoder inputs. On Windows use `php_onnxruntime.dll` for both commands. The directory is the one `make test` just loaded.

`make bench` measures one inference in-process against the same graph behind a Python HTTP service, and prints p50, p90, and p99 for both. It needs `bench/.venv`; the setup and the results are in [bench/README.md](bench/README.md).

### Homebrew

Apple Silicon. The Command Line Tools supply clang.

```bash
xcode-select --install
brew install php
export PATH="$(brew --prefix php)/bin:$PATH"
php -v
php-config --version
curl -fsSL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-osx-arm64-1.31.1.tgz
mkdir -p deps && tar -xzf ort.tgz -C deps
make test ORT="$PWD/deps/onnxruntime-osx-arm64-1.31.1"
```

### apt

Debian and Ubuntu. `php-dev` follows the distribution's default PHP, which must be 8.2–8.5.

```bash
sudo apt update
sudo apt install build-essential php-cli php-dev curl
php -v
php-config --version
curl -fsSL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz
mkdir -p deps && tar -xzf ort.tgz -C deps
make test ORT="$PWD/deps/onnxruntime-linux-x64-1.31.1"
```

On `arm64`, download `onnxruntime-linux-aarch64-1.31.1.tgz` instead. To compile against a specific minor version, install that pair, for example `php8.3-cli` and `php8.3-dev`, and put that `php` and `php-config` first on `PATH`.

### yum

RHEL, AlmaLinux, Rocky Linux, and Fedora. On Fedora the same packages install with `dnf`. `php-devel` must be PHP 8.2–8.5.

```bash
sudo yum install gcc make php-cli php-devel curl
php -v
php-config --version
curl -fsSL -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz
mkdir -p deps && tar -xzf ort.tgz -C deps
make test ORT="$PWD/deps/onnxruntime-linux-x64-1.31.1"
```

A ZTS library is a second build. Point `PATH` and `PHP_CONFIG` at the ZTS `php` and `php-config`, then run `make test` again. On Windows, put the ZTS `php.exe` first on `PATH` instead.

### Windows

Windows x64. PHP is a [windows.php.net](https://windows.php.net/) build, VS16 for 8.2 and 8.3, VS17 for 8.4 and 8.5. Visual Studio 2022 Build Tools with the C++ workload supplies `cl.exe`. Run the commands from Git Bash so `make` can see `uname`.

```bash
choco install make -y
export PATH="/c/php:$PATH"
php -v
curl -fsSL -o ort.zip https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-win-x64-1.31.1.zip
mkdir -p deps && unzip -q ort.zip -d deps
export PATH="$PWD/deps/onnxruntime-win-x64-1.31.1/lib:$PATH"
make test ORT="$PWD/deps/onnxruntime-win-x64-1.31.1"
```

Put the chosen PHP minor, NTS or ZTS, on `PATH` as `/c/php`. The machine also needs the Visual C++ Redistributable listed for that PHP build. `php_onnxruntime.dll` links `php8.dll` or `php8ts.dll`, `onnxruntime.dll`, and that runtime.

The extension version is the git tag without the leading `v`. A tag `v1.0.1` builds version `1.0.1`, and `php --ri onnxruntime` reports that. A commit after that tag keeps the distance, such as `1.0.1-3`, and does not include the commit id. With no tag in the clone, the build uses `0.1.0`. A release asset is `onnxruntime-1.0.1-php8.3-nts-linux-x64.tar.gz`: that version, the PHP minor, `nts` or `zts`, `linux`, `macos`, or `windows`, `x64` or `arm64`. Unpack it and keep the two files in the same directory. Linux has `onnxruntime.so` and `libonnxruntime.so.1`. macOS has `onnxruntime.so` and `libonnxruntime.1.dylib`. Windows has `php_onnxruntime.dll` and `onnxruntime.dll`. Load the extension:

```ini
extension=/path/onnxruntime.so
```

`extension=onnxruntime` looks for `onnxruntime.so` on Linux and macOS, and for `php_onnxruntime.dll` on Windows. On Windows the line is `extension=C:/path/php_onnxruntime.dll`. The file still has to match the PHP minor version, NTS or ZTS, and the CPU. A Linux build from Ubuntu 24.04 needs that glibc or a newer one. Windows builds are x64.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `make test` on every push and pull request: PHP 8.5 NTS on each runner. Each runner downloads the official ONNX Runtime 1.31.1 archive for that OS and CPU.

A `v*` tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml). That build is the full set: PHP 8.2–8.5, NTS and ZTS, on all four runners. Each archive contains the extension and the ONNX Runtime shared library. CI rejects a build that links any library besides that runtime and the system libraries.

## License

MIT, as in [LICENSE](LICENSE). The extension links the ONNX Runtime shared library, which is MIT too and is not bundled in the `.so`.
