"""g0d: derive the inverse SF-swizzle so we can build a PLAIN [N, K/16] scale tensor.
   Approach: inspect srcToDstBlk16RowMap + empirically map block->storage index."""
import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
dev="cuda"; torch.manual_seed(0)

def q16(x):
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),16,False)
    return fp4,sf,g

def main():
    print("srcToDstBlk16RowMap type:", type(fp4_utils.srcToDstBlk16RowMap),flush=True)
    m16=fp4_utils.srcToDstBlk16RowMap
    try:
        print("  len:", len(m16), "first 32:", list(m16[:32]),flush=True)
    except Exception as e:
        print("  ", e,flush=True)
    # get_shuffle_matrix_sf_a_row_indices signature
    import inspect
    try:
        print("get_shuffle_matrix_sf_a_row_indices:", inspect.signature(fp4_utils.get_shuffle_matrix_sf_a_row_indices),flush=True)
    except Exception as e: print(e,flush=True)
    print("get_shuffle_block_size:", inspect.getsource(fp4_utils.get_shuffle_block_size)[:300],flush=True)

    # EMPIRICAL inverse map: small N,K. Make each (row,block) scale identifiable.
    # Build w so that block (r,b) has a distinct magnitude. vec=16 blocks.
    N,K=128,64   # 4 blocks per row, 128 rows = 1 swizzle tile of 128 rows
    nb=K//16
    # set each block to a constant value c(r,b) so its computed scale is monotonic & distinct
    w=torch.zeros(N,K,device=dev,dtype=torch.bfloat16)
    for r in range(N):
        for b in range(nb):
            amax = 1.0 + (r*nb+b)*0.013  # distinct per block
            w[r, b*16:(b+1)*16] = amax
    wf,wsf,wg=q16(w)
    sf=wsf.view(torch.float8_e4m3fn).float()
    print(f"N={N} K={K} nb={nb} sf.numel={sf.numel()} (N*nb={N*nb})",flush=True)
    # The per-block scale that quantizer computes ~ amax_block * wg / 6 (fp4 max=6), rounded to fp8.
    # Expected scale for block(r,b): roughly (amax(r,b)*wg)/6. Build expected, find storage idx via nearest.
    exp = torch.zeros(N,nb,device=dev)
    for r in range(N):
        for b in range(nb):
            amax = 1.0 + (r*nb+b)*0.013
            exp[r,b] = amax*wg/6.0
    exp_flat = exp.flatten()  # plain order index = r*nb+b
    # For each plain index, find which storage position in sf matches (nearest in fp8-rounded space)
    # Round exp to fp8 to compare
    exp_fp8 = exp_flat.to(torch.float8_e4m3fn).float()
    sf_used = sf[:N*nb] if sf.numel()>=N*nb else sf
    # storage_for_plain[i] = argmin_j |sf_used[j]-exp_fp8[i]|
    diff = (sf_used.unsqueeze(0) - exp_fp8.unsqueeze(1)).abs()  # [Nplain, Nstore]
    storage_for_plain = diff.argmin(dim=1)
    # check uniqueness
    uniq = torch.unique(storage_for_plain).numel()
    print(f"unique storage targets: {uniq}/{N*nb} (1.0 means clean bijection)",flush=True)
    print("storage_for_plain[:16]:", storage_for_plain[:16].tolist(),flush=True)
    # Save the map shape info
    print("DONE",flush=True)

if __name__=="__main__": main()
