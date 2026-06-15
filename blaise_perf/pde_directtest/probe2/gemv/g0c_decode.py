"""g0c: nail the fp4 (E2M1) nibble decode + SF fp8 decode + plain-scale construction,
   so the custom GEMV can dequant correctly. Validate by reconstructing W in bf16 and
   comparing to the original, AND producing a 'plain scale' [N,K/16] that round-trips."""
import torch, tensorrt_llm
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
dev="cuda"; torch.manual_seed(0)

# E2M1 nibble (4-bit) -> float value table. Standard NVFP4 E2M1: sign(1) exp(2) mant(1).
# values: 0,0.5,1,1.5,2,3,4,6 and negatives.
E2M1 = torch.tensor([0.,0.5,1.,1.5,2.,3.,4.,6.,-0.,-0.5,-1.,-1.5,-2.,-3.,-4.,-6.], device=dev)

def q16(x):
    g=(448.0*6.0)/x.abs().max().float()
    fp4,sf=torch.ops.trtllm.fp4_quantize(x,g.reshape(1),16,False)
    return fp4,sf,g

def main():
    print(f"dev={torch.cuda.get_device_name(0)}",flush=True)
    N,K=256,512
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)
    wf,wsf,wg=q16(w)   # wf [N,K/2] uint8, wsf flat fp8(e4m3) swizzled, wg global scale
    print(f"wf {tuple(wf.shape)} {wf.dtype}; wsf {tuple(wsf.shape)} {wsf.dtype} numel={wsf.numel()} N*K/16={N*K//16}",flush=True)
    # decode nibbles: low nibble = element 2*j, high nibble = element 2*j+1
    lo = (wf & 0xF).long()
    hi = (wf >> 4).long()
    vals = torch.empty(N, K, device=dev)
    vals[:, 0::2] = E2M1[lo]
    vals[:, 1::2] = E2M1[hi]
    # Now need per-(row, kblock) scale. SF is swizzled. Use the SAME oracle path:
    # reference dequant = what fp4_gemm uses. Build ref via fp4_gemm with identity? Simpler:
    # reconstruct using torch's own understanding -> compare vals*scale to w.
    # The scale per block b (16 cols) decoded from fp8 e4m3, then /wg (global).
    # First find scale layout by brute force: try plain [N, K/16] row-major reading of wsf.
    sf_fp8 = wsf.view(torch.float8_e4m3fn).float()  # decode fp8 -> float
    print(f"sf_fp8 stats: min={sf_fp8.min():.4f} max={sf_fp8.max():.4f} mean={sf_fp8.mean():.4f}",flush=True)
    # Try interpret as plain [N, K//16]
    nb = K//16
    # Plain row-major candidate:
    if sf_fp8.numel()>=N*nb:
        cand_plain = sf_fp8[:N*nb].view(N, nb)
        rec_plain = vals.clone()
        rec_plain = rec_plain.view(N, nb, 16) * cand_plain.unsqueeze(-1)
        rec_plain = (rec_plain / wg).view(N,K).bfloat16()
        c = torch.nn.functional.cosine_similarity(rec_plain.float().flatten().unsqueeze(0), w.float().flatten().unsqueeze(0)).item()
        print(f"PLAIN row-major scale read: cos(recon, w) = {c:.5f}",flush=True)
    # Try the swizzle inverse via get_shuffle_matrix_sf_a_row_indices if available
    print("trying reswizzle_sf op for inverse...",flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
