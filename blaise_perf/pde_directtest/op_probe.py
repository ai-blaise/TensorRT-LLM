import torch, tensorrt_llm
print("TRTLLM", getattr(tensorrt_llm, "__version__", "?"))
ks = ["topk","sparse_mla","mqa_logits","indexer","dsa","hisa"]
ops = sorted([o for o in dir(torch.ops.trtllm) if any(k in o.lower() for k in ks)])
print("DECODE_OPS", ops)
for name in ops:
    op = getattr(torch.ops.trtllm, name, None)
    try:
        print("SCHEMA", name, "::", str(op.default._schema))
    except Exception as e:
        try:
            print("OVERLOADS", name, op.overloads())
        except Exception as e2:
            print("SCHEMA_ERR", name, repr(e), repr(e2))
