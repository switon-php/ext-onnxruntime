# AGENTS.md

`README.md` has the build steps. `stubs/README.md` is the public method list. `examples/` shows the call style. Review that style before adding a public method.

## Language

PHPDoc, READMEs, `phpinfo()` strings, and commit messages are English.

## API

The stubs are the surface. Change a public method there first. The extension links the official ONNX Runtime C API. Do not reimplement the runtime, and do not add a Rust or FFI wrapper in front of it.

Cover the common inference path. A short list may cross through `tensor()` or `values()` once. A large tensor stays on `fromBytes()` and `toBytes()`. `run()` returns `OnnxValue` handles and must not expand them. Do not add allocators, IOBinding, or other low-level controls to chase the last bit of speed.

A PHP string fed to a string-tensor input throws `OnnxException`. The message says a PHP string is packed bytes, and a string element belongs in a list or `OnnxValue::tensor()`.

## Tests

Every public method has a PHP assertion that shows its effect, including a success path. An error-only call does not count.

## Distribution

Packagist and PIE. Copy `composer.json` and the asset names in `.github/actions/test/action.yml`. The source build is `pie/config.m4`. Do not add PECL, Homebrew, or distro packages.
