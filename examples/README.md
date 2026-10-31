# Examples

Call-style samples for `Switon\OnnxRuntime`. Read them next to `stubs/`. They are not tests. From the repository root, `make test` builds the extension, then `make examples` writes `models/` and runs every script. `models/` holds small graphs with the same input and output names a service would use, so each file can be run. `embed.php` asks for `cuda` only when this build reports it, then `cpu`.

`embed.php` runs a batch of token ids and an attention mask, then mean-pools `last_hidden_state` in PHP. `classify.php` and `image.php` read `logits` and pick the label in PHP. `image.php` feeds a packed NCHW float buffer. `reuse.php` builds one `OnnxValue` and feeds it twice. `inspect.php` prints the inputs before any of that.
