#!/usr/bin/env python3
# Writes an F16 copy of a LoRA GGUF (every F32 tensor as F16, metadata kept): the non-trainable
# adapter learn-every-format.py --non-f32-refused loads. Usage: make-f16-shadow.py IN.gguf OUT.gguf
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "gguf-py"))
import numpy as np
import gguf

src, dst = sys.argv[1], sys.argv[2]
reader = gguf.GGUFReader(src)
arch = reader.fields["general.architecture"]
writer = gguf.GGUFWriter(dst, bytes(arch.parts[arch.data[0]]).decode())
for name, field in reader.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    kind = field.types[0]
    if kind == gguf.GGUFValueType.STRING:
        writer.add_string(name, bytes(field.parts[field.data[0]]).decode())
    elif kind == gguf.GGUFValueType.ARRAY:
        continue
    else:
        writer.add_key_value(name, field.parts[field.data[0]][0], kind)
for tensor in reader.tensors:
    data = np.array(tensor.data)
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        data = data.astype(np.float16)
    writer.add_tensor(tensor.name, data, raw_shape=list(reversed(tensor.shape.tolist())))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
