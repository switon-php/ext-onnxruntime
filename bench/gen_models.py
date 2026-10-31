#!/usr/bin/env python3
"""Write the graphs bench/run.php measures.

Three models, each answering one question:

  tiny.onnx       1x3 identity. How much does one run() cost before any math?
  encoder.onnx    6 x MatMul(256) over 1x64x256. A small encoder block's worth of work.
  bert.onnx       12 x MatMul(768) over 1x128x768. BERT-base's shape and arithmetic,
                  so the reader can see when the boundary stops mattering.
  classifier.onnx 1x256 -> 1x1000. The shape where PHP has no numpy, so the cost of
                  moving 1000 floats back into PHP is visible.

Weights are random but seeded, so every machine measures the same graph.
Run from the repository root:  bench/.venv/bin/python bench/gen_models.py
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = pathlib.Path(__file__).parent / "models"
SEED = 20261103
HIDDEN = 256
STEPS = 64
LAYERS = 6
CLASSES = 1000
BERT_HIDDEN = 768
BERT_STEPS = 128
BERT_LAYERS = 12


def save(nodes, inputs, outputs, initializers, path):
    graph = helper.make_graph(nodes, path.stem, inputs, outputs, initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="ext-onnxruntime bench",
    )
    # ONNX Runtime 1.31 accepts IR 9. A newer onnx package would otherwise write a
    # version that the linked runtime rejects.
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path.name}")


def tensor(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def weights(rng, name, shape):
    return numpy_helper.from_array(
        rng.standard_normal(shape).astype(np.float32) * 0.02, name
    )


def tiny():
    """Identity. The floor: no math, one run(), one value out."""
    node = helper.make_node("Identity", ["x"], ["y"])
    save([node], [tensor("x", ["batch", 3])], [tensor("y", ["batch", 3])], [], OUT / "tiny.onnx")


def stack(name, layers, hidden, steps, seed):
    """`layers` MatMul + Add + Relu blocks. BERT-base is this, plus attention."""
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    current = "x"
    for layer in range(layers):
        w, b = f"w{layer}", f"b{layer}"
        inits += [weights(rng, w, [hidden, hidden]), weights(rng, b, [hidden])]
        nodes.append(helper.make_node("MatMul", [current, w], [f"mm{layer}"]))
        nodes.append(helper.make_node("Add", [f"mm{layer}", b], [f"add{layer}"]))
        last = layer == layers - 1
        nodes.append(
            helper.make_node("Relu", [f"add{layer}"], ["y" if last else f"relu{layer}"])
        )
        current = "y" if last else f"relu{layer}"
    save(
        nodes,
        [tensor("x", ["batch", steps, hidden])],
        [tensor("y", ["batch", steps, hidden])],
        inits,
        OUT / name,
    )


def encoder():
    """Six blocks at 256. A small encoder, the size that fits in a sidecar service."""
    stack("encoder.onnx", LAYERS, HIDDEN, STEPS, SEED)


def bert():
    """Twelve blocks at 768 over 128 steps: BERT-base's arithmetic and memory shape."""
    stack("bert.onnx", BERT_LAYERS, BERT_HIDDEN, BERT_STEPS, SEED + 2)


def classifier():
    """One projection to 1000 classes. 4 kB of logits to move back into PHP."""
    rng = np.random.default_rng(SEED + 1)
    inits = [weights(rng, "w", [HIDDEN, CLASSES]), weights(rng, "b", [CLASSES])]
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["mm"]),
        helper.make_node("Add", ["mm", "b"], ["logits"]),
    ]
    save(
        nodes,
        [tensor("x", ["batch", HIDDEN])],
        [tensor("logits", ["batch", CLASSES])],
        inits,
        OUT / "classifier.onnx",
    )


if __name__ == "__main__":
    if not OUT.exists():
        OUT.mkdir()
    tiny()
    encoder()
    bert()
    classifier()
