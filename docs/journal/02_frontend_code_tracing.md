# Execution Log: TVM Python Frontend Tracing

This journal documents the exact code paths traced during our exploration of the TVM Python frontend, specifically analyzing how an ONNX model is ingested and translated into a TVM AST.

!!! info "Deep Dive"
    This execution log is organically linked to the theoretical discussion in [[02_frontend_parsing_and_ast]].

## Trace 1: The Entry Point (`from_onnx`)

When we run our Python script and invoke `tvm.relax.frontend.onnx.from_onnx()`, execution enters the Apache TVM codebase at `python/tvm/relax/frontend/onnx/onnx_frontend.py`.

```python
# python/tvm/relax/frontend/onnx/onnx_frontend.py (L5787)
def from_onnx(model, shape_dict=None, keep_params_in_input=False, ...):
    # ...
    g = ONNXGraphImporter(model, shape_dict, keep_params_in_input)
    graph_irmodule, graph_params = g.from_onnx(model, shape_dict)
    return graph_irmodule, graph_params
```

## Trace 2: Constructing Nodes (`ONNXGraphImporter`)

The `ONNXGraphImporter` object iterates through every node in the ONNX Protobuf graph. It maps the ONNX operator string (e.g., `"Conv"`) to its corresponding TVM parser class.

```python
# python/tvm/relax/frontend/onnx/onnx_frontend.py (L5988)
class ONNXGraphImporter:
    # ...
    def _construct_nodes(self, graph):
        for node in graph.node:
            op_name = node.op_type
            if op_name in self._renames:
                op_name = self._renames[op_name]
            
            # Retrieve the converter class for this specific operator
            convert_class = _get_convert_class(op_name)
            
            # Invoke the parser
            op = convert_class(self.bb, node, self.shape_dict, self.inputs_loc)
```

## Trace 3: The Operator Parser (`class Conv`)

When `op_name == "Conv"`, the converter class retrieved is `class Conv(OnnxOpConverter)`. This is where the actual transformation and "Desugaring" occurs.

```python
# python/tvm/relax/frontend/onnx/onnx_frontend.py (L1903)
class Conv(OnnxOpConverter):
    """Convert an onnx Conv node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        # 1. Determine dimension and operator (L1917)
        ndim = len(inputs[1].struct_info.shape)
        if ndim == 4:
            op = relax.op.nn.conv2d
            
        # 2. Emit pure convolution node (L1955)
        conv_out = bb.normalize(
            op(data=data, weight=inputs[1], ...)
        )
        
        # 3. Handle Bias via explicit Add node (L1967)
        if inputs[2] is not None:
            bias = relax.op.reshape(inputs[2], ...)
            conv_out = relax.op.add(conv_out, bias)

        return conv_out
```

### Key Finding
Through this code trace, we observed firsthand how the high-level API breaks down convenience structures (like combined Conv+Bias) into distinct, atomic memory representations (`AST Nodes`) using the `BlockBuilder` (`bb`).
