import contextlib


def patch_torch_dynamo_transformers_compat():
    try:
        import torch._dynamo._trace_wrapped_higher_order_op as dynamo_hop
    except Exception:
        return

    if hasattr(dynamo_hop, "TransformGetItemToIndex"):
        return

    class TransformGetItemToIndex(contextlib.AbstractContextManager):
        def __exit__(self, exc_type, exc, tb):
            return False

    dynamo_hop.TransformGetItemToIndex = TransformGetItemToIndex


patch_torch_dynamo_transformers_compat()
