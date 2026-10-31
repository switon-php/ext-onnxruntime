# Benchmark

What one inference costs when the model lives in the PHP process, and what it costs
when the model lives in a Python process behind HTTP.

The question this answers: *why not just keep the Python service?* The answer depends
on the model. This measures both sides on the same machine, the same ONNX Runtime, the
same thread counts, in the same second.

## Run it

```bash
python3 -m venv bench/.venv
bench/.venv/bin/pip install -r bench/requirements.txt

make bench
```

`make bench` builds the extension, writes the graphs, then measures. No model files are
stored in the repository; `bench/gen_models.py` writes them and they are gitignored.

The Python side is pinned to `onnxruntime==1.31.1`, the version the extension links, so
the two runs are not comparing different kernels.

## What is measured

Four graphs, each answering one question.

| Graph | Shape | Question |
| --- | --- | --- |
| `tiny.onnx` | 1x3 identity | What does one call cost before any math? This is the floor. |
| `encoder.onnx` | 6 x MatMul(256) over 1x64x256 | A small model, the size that fits in a sidecar service. |
| `bert.onnx` | 12 x MatMul(768) over 1x128x768 | BERT-base's arithmetic and memory shape, so you can see when the boundary stops mattering. |
| `classifier.onnx` | 1x256 -> 1x1000 | The shape where PHP has no numpy: 1000 floats to bring back. |

Both sides use `intra_op_num_threads=4`, `inter_op_num_threads=1`, sequential
execution, and `ORT_ENABLE_ALL`. Both keep one session and one connection. Both are
timed per call with `hrtime()`, after a warmup, and every sample is kept.

The Python baseline is deliberately plain: stdlib `http.server`, HTTP/1.1 keep-alive,
raw float32 in and out, no framework, no JSON. A Flask or FastAPI service would add its
own overhead on top, which would flatter the extension. This is the charitable version.

## Results

Apple M2, 8 cores, PHP 8.5.3, ONNX Runtime 1.31.1, `intra_op=4`. Milliseconds. Each
row is the **median of five runs** of the whole script, 200 iterations after 30 warmup
per run (100 after 10 for `bert.onnx`). A single run drifts by up to 20% on a laptop, so
one run is not enough to publish a number.

### `tiny.onnx` 1x3 - fixed cost of one call

| Path | p50 | p99 |
| --- | ---: | ---: |
| extension `run()` | 0.003 | 0.003 |
| Python over HTTP | 0.16 | 0.25 |

The floor of the boundary is **0.15 ms** on loopback, with no TLS, no framework, no
authentication, and a connection that is already open.

### `encoder.onnx` 1x64x256 - a small model

| Path | p50 | p99 |
| --- | ---: | ---: |
| extension, PHP array feed -> `toBytes()` | 0.26 | 0.59 |
| extension, `fromBytes()` feed -> `toBytes()` | 0.21 | 0.40 |
| Python over HTTP, packed body -> raw bytes | 0.38 | 0.95 |
| `pack('f*')` of the same 16384 floats in PHP | 0.09 | 0.14 |

At this size the boundary adds 0.17 ms to a 0.21 ms call, so it nearly doubles it. Note
the last row: if the data starts as a PHP array, the HTTP path also has to pack it, and
the extension path does not.

### `bert.onnx` 1x128x768 - BERT-base arithmetic

| Path | p50 | p99 |
| --- | ---: | ---: |
| extension, PHP array feed -> `toBytes()` | 6.51 | 9.48 |
| extension, `fromBytes()` feed -> `toBytes()` | 6.25 | 11.58 |
| Python over HTTP, packed body -> raw bytes | 6.69 | 8.05 |

At this size the boundary adds 0.44 ms to a 6.25 ms call, about 7%, and the model
dominates. If your model looks like this, the Python service is not the thing to
optimize.

### `classifier.onnx` 1x256 -> 1x1000 - bringing 1000 floats back

| Path | p50 | p99 |
| --- | ---: | ---: |
| extension, `values()` (1000 PHP floats) | 0.021 | 0.057 |
| extension, `toBytes()` (4 kB) | 0.019 | 0.116 |
| extension, `values()` then a PHP argmax loop | 0.028 | 0.067 |
| extension, `['index' => 'max']` | 0.013 | 0.018 |
| Python over HTTP, argmax on its side | 0.18 | 0.44 |

The output-request form saves about 15 microseconds over materializing 1000 floats and
looping in PHP. That is real, but it is small: **the reason to use it is that you do not
write the loop and do not allocate 1000 zvals, not that it is fast.** The 14x gap in
this table is the HTTP boundary again, not the argmax.

## What this does and does not say

It says: the cost of keeping the model in Python, on loopback, with everything warm, is
**about 0.15 ms plus roughly half a microsecond per kilobyte moved each way**.

It does not say the extension makes inference faster. Both sides run the same ONNX
Runtime on the same graph. The extension removes a boundary; it does not speed up math.

Three caveats worth stating:

- **The graphs are synthetic.** Random weights, stacked MatMul. `bert.onnx` has BERT's
  arithmetic and memory shape but no attention. Absolute times will not match a real
  exported model. The difference between the two paths is what this measures, and that
  difference does not depend on the weights.
- **One machine, one sitting, a laptop.** What is published is the median of five
  runs, because a single run drifts by up to 20% here. Expect thermal and scheduler
  noise in the p99, which is why the tables report both. Absolute times belong to this
  machine; the gap between the two columns is the part that travels.
- **A real service is worse than this baseline**, not better: TLS, JSON or multipart,
  a framework, auth, and a queue all add to the HTTP column.

If your shell exports `http_proxy`, `bench/run.php` disables it for the loopback calls.
Without that, curl routes through the proxy and the request line carries an absolute
URI, which both inflates the numbers and breaks the server's routing.
