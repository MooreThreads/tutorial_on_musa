import torch
import torch_musa
import time
from torch.nn import functional as F


def make_causal_4d_mask_float(
    input_ids_shape, dtype: torch.dtype, device: torch.device = torch.device("cpu")
):
    """
    Make Casual 4D float mask
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


def make_causal_4d_mask_bool(
    input_ids_shape,
    device: torch.device = torch.device("cpu"),
):
    """
    Make Casual 4D bool mask
    """

    bsz, tgt_len = input_ids_shape
    mask = torch.tril(torch.ones((bsz, tgt_len, tgt_len), device=device)).view(
        bsz, 1, tgt_len, tgt_len
    )
    mask = mask > 0.5

    return mask


def generate_square_subsequent_mask(seq_len: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def generate_pad_mask(batch_size: int, seq_len: int):
    mask = torch.zeros([batch_size, seq_len], dtype=torch.float)
    for b in range(batch_size):
        r = torch.randint(1, seq_len - 1, (1,))
        mask[b][-r:] = -torch.inf
    return mask


def gen_input_data(case, mask_type, dtype=torch.float32):
    """
    Generating the mocked input data of SDP.
    """
    item = {}

    kv_num_heads = case[-1]
    num_heads = q_num_heads = case[-2]
    emb_dim = case[-3]
    is_gqa = q_num_heads != kv_num_heads
    if is_gqa:
        assert (
            q_num_heads % kv_num_heads == 0
        ), "Query's head_num should be evenly divided by key/value's head_num."
    assert emb_dim % num_heads == 0  # emb_dim must be evenly divided by num_heads
    head_dim = emb_dim // num_heads
    total_shape = case[0]
    batch_size = total_shape[0]
    seq_len = total_shape[1]
    if not is_gqa:
        total_shape = (batch_size, num_heads, seq_len, 3 * head_dim)
        qkv = torch.randn(total_shape, dtype=dtype)
        # q,k,v has the same shape: [B, num_heads, T, head_dim]
        query, key, value = qkv.chunk(3, -1)  # pylint: disable=invalid-name
    else:
        query = torch.randn([batch_size, q_num_heads, seq_len, head_dim], dtype=dtype)
        key = torch.randn([batch_size, kv_num_heads, seq_len, head_dim], dtype=dtype)
        value = torch.randn([batch_size, kv_num_heads, seq_len, head_dim], dtype=dtype)

    item["query"] = query
    item["key"] = key
    item["value"] = value

    # generating bool mask.
    if mask_type == 1:
        # padding mask
        mask = generate_pad_mask(batch_size, seq_len)
    elif mask_type == 0:
        # key padding
        mask = generate_square_subsequent_mask(seq_len)
    elif mask_type == 2:
        mask = make_causal_4d_mask_float((batch_size, seq_len), dtype=dtype)
    elif mask_type == 3:
        mask = make_causal_4d_mask_bool((batch_size, seq_len))
    elif mask_type == 4:
        mask = ~make_causal_4d_mask_bool((batch_size, seq_len))
    else:
        mask = None

    if mask is not None and mask.dtype != torch.bool:
        mask = mask.to(query.dtype)

    item["attn_mask"] = mask

    return item


if __name__ == "__main__":
    is_causal = False
    # [(bs, seq_len, embedding_dim), embedding_dim, q_head_num, kv_head_num]
    case = [(256, 256, 1024), 1024, 16, 16]
    dtype = torch.half
    # MASK_TYPES = [-1, 0, 1, 2]
    mask_type = 1
    print("================================================================")
    print("==================SDP kernel by FlashAttention==================")
    print("================================================================")
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True):
            input_data = gen_input_data(case, mask_type, dtype)

            query = input_data["query"].musa()
            key = input_data["key"].musa()
            value = input_data["value"].musa()
            attn_mask = input_data["attn_mask"].musa()
            dropout_p = 0.0
            batch_size, _, seq_len, _ = query.shape
            if (
                attn_mask is not None
                and attn_mask.shape == (batch_size, seq_len)
                and attn_mask.is_cpu
            ):
                # we should make the mask broadcastable to the atten_probs
                attn_mask = attn_mask.view(batch_size, 1, 1, seq_len)
            # =====warmup start=====
            for i in range(5):
                F.scaled_dot_product_attention(
                    query, key, value, attn_mask, dropout_p, is_causal
                )
            # =====warmup end=====

            torch.musa.synchronize()
            start_time = time.perf_counter()
            NUMBER = 10
            for i in range(NUMBER):
                F.scaled_dot_product_attention(
                    query, key, value, attn_mask, dropout_p, is_causal
                )
            torch.musa.synchronize()
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) / NUMBER
            print("dtype: ", dtype)
            print("query.shape: ", query.shape)
            print("key.shape: ", key.shape)
            print("value.shape: ", value.shape)
            print("attn_mask.shape: ", attn_mask.shape)
            print("dropout_p: ", dropout_p)
            print("is_causal: ", is_causal)
            print("mask_type: ", mask_type)
            print("===elapsed_time: {} ms".format(elapsed_time * 1000))
