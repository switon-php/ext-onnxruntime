#!/usr/bin/env python3
"""The baseline bench/run.php measures against: a Python ONNX Runtime behind HTTP.

This is the shape most PHP services use today. One process holds the session, the
PHP side posts a tensor and gets a tensor back. The point of the measurement is the
cost of that boundary, not the cost of Python.

Deliberately plain: stdlib http.server, no framework, HTTP/1.1 keep-alive, no JSON.
A Flask or FastAPI service would add its own overhead on top, which would flatter
the extension. This is the charitable version of the baseline.

  bench/.venv/bin/python bench/server.py --port 8731 --threads 4

Endpoints:
  POST /run/<model>     body is raw little-endian float32, reply is raw float32
  POST /argmax/<model>  reply is one little-endian int32 per row
  GET  /health          reply is the loaded model names
"""

import argparse
import http.server
import pathlib
import sys

import numpy as np
import onnxruntime as ort

MODELS = pathlib.Path(__file__).parent / "models"
SHAPES = {
    "tiny": (1, 3),
    "encoder": (1, 64, 256),
    "bert": (1, 128, 768),
    "classifier": (1, 256),
}
SESSIONS = {}


def load(threads):
    options = ort.SessionOptions()
    # Pinned so both sides of the benchmark use the same thread pool. ORT's default
    # is one thread per physical core, which would make the two runs incomparable on
    # a machine whose core count differs from the one this was measured on.
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for name in SHAPES:
        SESSIONS[name] = ort.InferenceSession(
            str(MODELS / f"{name}.onnx"), options, providers=["CPUExecutionProvider"]
        )


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # a per-request log line would dominate the measurement

    def reply(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(",".join(SESSIONS).encode())
            return
        self.send_error(404)

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if len(parts) != 2 or parts[1] not in SESSIONS:
            self.send_error(404)
            return
        mode, name = parts
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        session = SESSIONS[name]
        shape = SHAPES[name]

        tensor = np.frombuffer(body, dtype="<f4").reshape(shape)
        output = session.run(None, {"x": tensor})[0]

        if mode == "run":
            self.reply(output.astype("<f4", copy=False).tobytes())
        elif mode == "argmax":
            self.reply(np.argmax(output, axis=-1).astype("<i4").tobytes())
        else:
            self.send_error(404)


class Server(http.server.HTTPServer):
    # Single-threaded on purpose. The benchmark sends one request at a time, so a
    # thread per connection buys nothing and makes the framing harder to reason about.
    allow_reuse_address = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    load(args.threads)
    server = Server(("127.0.0.1", args.port), Handler)
    print(f"listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
