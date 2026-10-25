# Kernel sources of the REAX s1-fast engine committed on SN92 UID 220

Entry: UID 220, hotkey `5F2E7wLRvwfNKxtACmH5AF1kELhYE8po7u1jH91ov5eYdU6A`, competition element
`reax-eng/1:s1-fast:ghcr.io/johnkell9801-hub/s1q9@sha256:f1422b06f6098ad7b8939c1e05a48b82493d2ccc672912b92e21491256ca0898`.

Every compiled part the image ships is in `/opt/s1eng/s1eng/` (43 files: 42 CUDA cubins and one host
library). Each one rebuilds **byte for byte** from the sources in this folder with the build line below, run in
`/opt/s1eng/s1eng`. The image also ships each source next to its binary, and `/opt/s1eng/s1eng/KERNELS.md` lists the same
binaries with the same hashes and build lines. 5 of the 43 binaries are new since UID 240's entry (sources published in
[reax-uid240-kernels](https://github.com/johnkell9801-hub/reax-uid240-kernels), which also lists everything published
with UIDs 164 and 223); the other 38 are byte-identical to files published there and are listed again so
this folder is complete.

Layers: the kernel files sit in the image's top layers below. The final copy of each binary is in layer 20 (39 binaries),
layer 21 (`q8k.cubin`) or layer 22 (`q8pb.cubin`, `q8v.cubin`, `q8wt.cubin`); layers 16, 17 and 19 hold earlier copies of
some of them, each byte-identical to its final copy (no shadowed binary differs). Layer 18 holds only notes (no binary).
One source is shadowed: layer 21 replaces layer 20's `q8.cu` (`16511f99336015f3…`) with the `q8.cu` in this folder, which
only appends the `#ifdef Q8K` block that `q8k.cu` compiles (no line above it changed); `q8.cubin` rebuilds byte for byte
from this folder's `q8.cu`.

* layer 16: `sha256:184d097c857b374eb0f6d175159d810178ec754667d6f73b871130574a021970`
* layer 17: `sha256:033c5dc751ab17bd19decf5ef83f6ef36d431a0adb8862aebc120390e80a30f2`
* layer 19: `sha256:acb61750b3ebe6bbf9916f67c7a1d1798277e3fd5b5f4a3e4a578ba7f86e82c2`
* layer 20: `sha256:b2d1b2849e8291a794267ed29e97bdf856661f7c5191d7de194fb4a09b336f72`
* layer 21: `sha256:849ddd5f796b4e0ef3282a1e0186f33dc8ee8d3fa105281f8c7618fcb6ca5a72`
* layer 22: `sha256:77733d5949dc2d7730f890465b8d816db3c81e43f77053c32a4a491d3de6c7ef`

| binary (layers) | sha256 | source in this folder (sha256) | build line | toolkit | author | new since UID 240 |
|---|---|---|---|---|---|---|
| `ngemm5.cubin` (16, 19, 20) | `7e7775b0f52bf73c5848d5d0016070663ec1c0581389b48d06bb2a4d29585b94` | `ngemm5.cu` (`1d6410833144ece2…`) | `nvcc -arch=sm_120a -O3 -cubin -o ngemm5.cubin ngemm5.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `ngemm9b.cubin` (16, 19, 20) | `f097e26cfa1064f3e784ce0bb6bf79263491761933f4cd9d9f1ec9ef67545937` | `ngemm9b.cu` (`535c9b97c52d0e9d…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9b.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `ngemm9b_wn.cubin` (20) | `a500d86659c59a9eb0928b82bac78073a2b4b5d0b5cc165689d12cef72914373` | `ngemm9b.cu` (`535c9b97c52d0e9d…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngemm9b_wn.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 240) |
| `ngemm9b_wnt.cubin` (20) | `5c33f79864181ea19d529e70d533303840831452c806c8b1907fa8f836e6a773` | `ngemm9b.cu` (`535c9b97c52d0e9d…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngemm9b_wnt.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 240) |
| `ngemm9t.cubin` (16, 19, 20) | `111b70b04ac63f4c12f82145d45a1ae64bafb3c5f812fd3dae4b195b08e7f70d` | `ngemm9t.cu` (`4af699b4b0b4ad0b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9t.cubin ngemm9t.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `ngemm9t_wn.cubin` (20) | `daf373f578a3dfe87ab9bad1d4673ddb6af9bc2bb0e1c99593f284e23a07955b` | `ngemm9t.cu` (`4af699b4b0b4ad0b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngemm9t_wn.cubin ngemm9t.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 240) |
| `ngemm9t_wnt.cubin` (20) | `a5e3562bbd7ed512d55f51aaf872f4a7a0b3523a578845b888dc8882f79be142` | `ngemm9t.cu` (`4af699b4b0b4ad0b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngemm9t_wnt.cubin ngemm9t.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 240) |
| `ngemm9l.cubin` (16, 19, 20) | `f35d8d2a2d44d01a0de8154adeaac94651035aaf83aaf04e021f048d37b57e88` | `ngemm9l.cu` (`27a5d2f0eed18ec7…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9l.cubin ngemm9l.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `ngr.cubin` (16, 19, 20) | `15dc8936691b884e1364a5c57f64470e354bc59c6c64c1244307fcf614f8e889` | `ngr.cu` (`7831294a4fa20335…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngr.cubin ngr.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `ngr_wn.cubin` (20) | `0ecbf8d81df54457d451d357a03f1ef696ef869755c3d402b749482a10ee3fd1` | `ngr.cu` (`7831294a4fa20335…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngr_wn.cubin ngr.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX-R, native rewrite 10-24: EPX's tiled-weight addresses + n-major issue order on RARN's GEMMs) | no (published with UID 240) |
| `xa.cubin` (16, 19, 20) | `8d119f6b271de9ccf86d58b99b5a37299cc9460b1c247f4a74e70bd8154762aa` | `xa.cu` (`231ea2f7bfd427b1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o xa.cubin xa.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `kkt2n.cubin` (16, 19, 20) | `fd5b50c85669d3f106d3415523b62651bf5e6ace69cb0531404059ba87000c3a` | `kkt2n.cu` (`05f7c62710f2e8ba…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o kkt2n.cubin kkt2n.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `ffs.cubin` (16, 19, 20) | `3c739cd2bfef61f8199601b4122dd4cda971c90015a4165284c696dea250c8f2` | `ffs.cu` (`724abb42dcca68d9…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ffs.cubin ffs.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `pron.cubin` (16, 19, 20) | `f34ec4fb457a129b920928ea3355e689cd739a77d94b75e6236da5e053a34429` | `pron.cu` (`8bfa8dd5f059d4f9…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DPRON_SHIP -o pron.cubin pron.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 240) |
| `onorm.cubin` (16, 19, 20) | `e1188ab018788db45f5630f2c8fb7dee62c120a86f8aea072846efecedf86b8c` | `onorm.cu` (`ec42b12a08a1278d…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onorm.cubin onorm.cu` | CUDA 13.4 (`V13.4.92`) | rgh (source onorm_v17.cu) | no (published with UID 240) |
| `kf.cubin` (16, 19, 20) | `63117c351394d8e3bf6635bde4fffbe9e5e6aff03d2aa682eeee05b102608a0c` | `kf.cu` (`ea0ebf9ab3bb6da1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o kf.cubin kf.cu` | CUDA 13.4 (`V13.4.92`) | ruj (the `KF_NO_KERNELS` guard and the `late` hook in `kf.cu`: kai) | no (published with UID 240) |
| `kf_wn.cubin` (20) | `da54a1b42746ad55740516daf39f66a6a8a884442a65a6ea4efbb156977d066a` | `kf.cu` (`ea0ebf9ab3bb6da1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DKF_WTL -DKF_NMAJ -o kf_wn.cubin kf.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 240) |
| `kf_wnt.cubin` (20) | `37439958d3092a8dbd14d5289e395fd85f5476c326e91952345e2b8133b2a700` | `kf.cu` (`ea0ebf9ab3bb6da1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DKF_WTL -DKF_NMAJ -DTROW -o kf_wnt.cubin kf.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 240) |
| `skx6.cubin` (16, 19, 20) | `d5e3b11dbf53e0b5fa08bbd633d685f64e63738fdb00cd7777dd54176b83caf5` | `skx6.cu` (`4f0e74d32b6ebabf…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6.cubin skx6.cu` | CUDA 13.4 (`V13.4.92`) | ruj | no (published with UID 240) |
| `skx6t.cubin` (16, 19, 20) | `017ca4f7f600affcd2cfb3659c98ccb26ca04e3f65993f491e57dc9cd349ca75` | `skx6t.cu` (`d64594a7372f2c39…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6t.cubin skx6t.cu` | CUDA 13.4 (`V13.4.92`) | ruj | no (published with UID 240) |
| `rn.cubin` (16, 19, 20) | `88247357dccfa88bd1d7c10eac47e3eb7e079e10498b30daacc3c5987b467487` | `rn.cu` (`8c00eac6f0c4170d…`) | `nvcc -cubin -arch=sm_120a -O3 -fmad=false -Xptxas -v -o rn.cubin rn.cu` | CUDA 13.4 (`V13.4.92`) | ruj | no (published with UID 240) |
| `ngemm7.cubin` (16, 19, 20) | `8671e9a9bec1573a9c5ffe94d4ace7475f38509a77e9cec346ccee79246a827f` | `ngemm7.cu` (`96ff118ce403cbab…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -o ngemm7.cubin ngemm7.cu` | CUDA 13.4 (`V13.4.92`) | pjs (nsw7l, ported by rgh) | no (published with UID 240) |
| `kfx.cubin` (16, 19, 20) | `8ed921f761f1632bf08a4637b4c018fe059964cf468e8303d1899dec41c992fb` | `kfx.cu` (`42529f1d88bcccb5…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o kfx.cubin kfx.cu` | CUDA 13.4 (`V13.4.92`) | kai (instantiates ruj's KF tile; includes `kf.cu` and `kfx_map.h`) | no (published with UID 240) |
| `zcin.cubin` (16, 19, 20) | `f2ea51229214c944e873777b2f34812b3a6ef343686674564927331a98793133` | `zcin.cu` (`3e94fa1acd1e5695…`) | `nvcc -cubin -arch=sm_120a -O3 -Xptxas -v -o zcin.cubin zcin.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 240) |
| `xa3.cubin` (16, 19, 20) | `31cd78d9e28dc0ced0ae3f8283fe3ad2d8604bd5b5af159f220b9e903c612032` | `xa3.cu` (`bf836985bd3af8c7…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o xa3.cubin xa3.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 240) |
| `onormpv.cubin` (16, 19, 20) | `b4a29ab178559c8b8a9f8146e5ecfdc45dee9e6383dc0e6d2375f58d6e32e7b0` | `onormpv.cu` (`24b7bbdcb7098289…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onormpv.cubin onormpv.cu` | CUDA 13.4 (`V13.4.92`) | ruj (PROFUSE-v on ONORM; includes rgh's `onorm.cu` unchanged; the v staging after rgh's dev_profuse ffsp.cu) | no (published with UID 240) |
| `ngemm9b_trow.cubin` (17, 19, 20) | `08c60e917d5bbf31d77db1ea42fadf27b2ae25b303f2ea9ddfd2bffff3413e7b` | `ngemm9b.cu` (`535c9b97c52d0e9d…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngemm9b_trow.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 240) |
| `ngemm9t_trow.cubin` (17, 19, 20) | `789b913ddf4ed3eefacfaaafefb47fd284333ca43c1110852d5609939971ad4c` | `ngemm9t.cu` (`4af699b4b0b4ad0b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngemm9t_trow.cubin ngemm9t.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 240) |
| `ngemm7_trow.cubin` (17, 19, 20) | `a444dd2d2b11181c283a67baeaffe14901227438734067401b3c71ae090404ee` | `ngemm7.cu` (`96ff118ce403cbab…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -DTROW -o ngemm7_trow.cubin ngemm7.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 240) |
| `ngr_trow.cubin` (17, 19, 20) | `6216520edc39d60ea9c5c4d1355585fac30202137e132a4b2349802dc787be09` | `ngr.cu` (`7831294a4fa20335…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngr_trow.cubin ngr.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 240) |
| `ngr_wnt.cubin` (20) | `913ca3a1df92bcf81599025e01377a153ce35180b3af17ce746ff705690ec113` | `ngr.cu` (`7831294a4fa20335…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngr_wnt.cubin ngr.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX-R, native rewrite 10-24: EPX's tiled-weight addresses + n-major issue order on RARN's GEMMs) | no (published with UID 240) |
| `kf_trow.cubin` (17, 19, 20) | `00e5eacf188344da970d8c211eb027bf3200edaeec0c1a5d5ab6bda6a94e8906` | `kf.cu` (`ea0ebf9ab3bb6da1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o kf_trow.cubin kf.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 240) |
| `kfx_trow.cubin` (17, 19, 20) | `608e98e16f2077ef5ae143c8af69093266c8b21e50ace28572efe9c1aa17fa68` | `kfx.cu` (`42529f1d88bcccb5…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -DTROW -o kfx_trow.cubin kfx.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 240) |
| `ngemm8.cubin` (17, 19, 20) | `b560f9354f31354d33ad3227082b2b51e3ec235d917b78fc392cc23652519c1e` | `ngemm8.cu` (`fcbd047c076899b2…`) | `nvcc -O3 -arch=sm_120a -cubin -o ngemm8.cubin ngemm8.cu` | CUDA 13.4 (`V13.4.92`) | pjs (NG8, e17 / e20 / e22p: NG7's mainloop / epilogue / arithmetic with 192 x 192 body tiles; its t160 / t224 bands follow rgh's NG9T idea) | no (published with UID 240) |
| `ngemm7n.cubin` (19, 20) | `7fdb6980d653f6e4425cf51ee09d14aca0c1d7e63fd53874199a318b20ff88ed` | `ngemm7n.cu` (`4ceda9b976285ecd…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -o ngemm7n.cubin ngemm7n.cu` | CUDA 13.4 (`V13.4.92`) | pjs (e27 NMAJ: ngemm7.cu with an n-major HMMA issue order, kernels `*_n`) | no (published with UID 240) |
| `ngemm7w.cubin` (19, 20) | `c3a4f7d88a1247519b508d5f1a3f78dce78bf317d72a2126de11a60678b3a34d` | `ngemm7w.cu` (`373e8af2d43090a8…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -DNG7_WTL -o ngemm7w.cubin ngemm7w.cu` | CUDA 13.4 (`V13.4.92`) | pjs (WTL twins of the nsw7l kernels incl. rgh's four NG7T tall-tail instantiations) | no (published with UID 240) |
| `ngemm7wn.cubin` (19, 20) | `c2b076f768537061397cf6f401a02e66df67f4a42eef0a168eab5c68c1a14610` | `ngemm7wn.cu` (`da669583828fea3d…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -DNG7_WTL -o ngemm7wn.cubin ngemm7wn.cu` | CUDA 13.4 (`V13.4.92`) | pjs (e27-BUILD WN: ngemm7w.cu with NMAJ's n-major k16 issue order, kernels `*_wn`; incl. rgh's four NG7T tall-tail instantiations) | no (published with UID 240) |
| `q8.cubin` (20) | `917aab0ee17f08dc16b57f81e1e721bae7c4b0480d7efb8d6fe1a723feff45f7` | `q8.cu` (`f7dcf568658de60b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8.cubin q8.cu` | CUDA 13.4 (`V13.4.92`) | rgh (Q8, native rewrite 10-24: W8A8 int8 projections, fine-grained down / out groups, fused SwiGLU -> int8 and add + RMSNorm -> int8; ROT: 128-point Hadamard rotation of the fine groups, q8gr_* / q8x_prepfr) | yes |
| `q8k.cubin` (21) | `e022f80b3fbb9c204a8399199ebbcc3f0f677de41a0a16cc07fdbe2ec3b86e64` | `q8k.cu` (`30d05fbe9f38683c…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8k.cubin q8k.cu` | CUDA 13.4 (`V13.4.92`) | ruj (r8q2 Q8K twins of rgh's Q8 kernels, ported onto g9r6 with twins of rgh's ROT kernels q8gr_* (g9x); `q8k.cu` = `#define Q8K 1` + `#include "q8.cu"`: the `Q8K` block of `q8.cu`) | yes |
| `q8wt.cubin` (22) | `e4a317751b91e2d708789ed014e85523ec7a480d9f9f428f6fdc032a09a372f9` | `q8wt.cu` (`ceb2c0602fe7b852…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8wt.cubin q8wt.cu` | CUDA 13.4 (`V13.4.92`) | pjs (Q8-WT, e33: twins of rgh's q8h_ / q8g_ / q8f_ kernels reading a tile-contiguous copy of the int8 weights; q8wt.cu = rgh's q8.cu with only the int8 B cp.async source addresses changed) | yes |
| `q8v.cubin` (22) | `1ee120ece5783b6e84939b7d7bbc56a8a8a6272ae054bec8a0dd91124b6b6aa4` | `q8v.cu` (`fa4a4ab886990fd9…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8v.cubin q8v.cu` | CUDA 13.4 (`V13.4.92`) | pjs (Q8-V, e35: exact schedule variants of rgh's q8h_ / q8g_ / q8gr_ kernels and of pjs's Q8-WT q8h twins: C1 = the int8 main loop and the protected bf16 pass in one cp.async ring; C1T = C1 + the SwiGLU epilogue's silu read from a 65,536-entry table built with rgh's swiglu1 (q8v_silu_tab); q8v.cu = rgh's j206 q8.cu namespace text + q8wt.cu's Loader TB edit + the variants) | yes |
| `q8pb.cubin` (22) | `9d85f48101785b5158519b20eeda98aeb864710ed8907598da54242dbc0402c6` | `q8pb.cu` (`294b1e481d000cbd…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8pb.cubin q8pb.cu` | CUDA 13.4 (`V13.4.92`) | pjs (Q8-PB / Q8-NX, e35: twins of rgh's q8f_ fine-group kernels whose epilogue stores bf16 rows - _pbd: bf16x2 register stores, _pb: rgh's store_tile - instead of the fp32 partial, and rgh's q8n_addnorm with a null-x option (q8n_addnorm_nx); q8pb.cu = rgh's j206 q8.cu namespace + tile_fpb / hetero_fpb / store_rows_bf16 + q8n_addnorm_nx) | yes |
| `ftokc.so` (16, 19, 20) | `2b34d5b4b7b56dfbe38fa57be16ac1005da54b1b64ce886cdb02bc0ea0e224dc` | `ftokc.c` (`1fe352dc6cf51083…`) | `gcc -O2 -fPIC -shared -nostdlib -ffreestanding -fno-builtin -o ftokc.so ftokc.c` | gcc 13.3.0 (Ubuntu 24.04) | kai | no (published with UID 240) |

Nothing else in the engine layers is compiled: the other files are Python source, JSON tables, notes (`KERNELS.md`, lever
notes) and the Q8 weight / calibration data (`q8_gptq.pt`, `q8_calib.pt`). Layers 1-15 are the official
`ghcr.io/reaxlabs/reax-s1-fast-runtime:1.2.0-rc1` layers, unchanged.

## What is new

The new binaries against UID 240 (every other one is byte-identical to its published file):

* `q8.cubin` (`q8.cu`, rgh): ROT - a randomised 128-point Hadamard rotation of the down / out projections' inputs, applied
  in the gate|up + SwiGLU epilogue (`q8gr_*`: the tile's 128 SwiGLU outputs per row rotated by one warp through shared
  memory before the int8 rounding) and in the out_proj prepare (`q8x_prepfr`); the int8 weights are stored rotated the
  same way (GPTQ on rotated Hessians, `q8_gptq.pt`). Q8 runs from layer 6.
* `q8k.cubin` (`q8k.cu`, ruj): bit-identical schedule twins of the Q8 GEMMs (LUT / S96 / ring / WTL tiled weights).
* `q8pb.cubin` (`q8pb.cu`, pjs): Q8-PB / Q8-NX - the fine int8 projections write bf16 rows that `q8n_addnorm` reads.
* `q8v.cubin` (`q8v.cu`, pjs) and `q8wt.cubin` (`q8wt.cu`, pjs): Q8-V / Q8-WT schedule twins, shipped in the image
  (switched off in this entry's environment).

## Notes

* `-Xptxas -v` matters where a build line has it: ptxas writes its own command line into the cubin's `.note.nv.tkinfo`.
* `onorm.cubin` and `onormpv.cubin`: `onorm.cu` (also included by `onormpv.cu`) has a device function in an anonymous
  namespace, whose internal name includes the CRC32 of the source's absolute path, so they are byte-identical only when built
  from `/opt/s1eng/s1eng/`. Built elsewhere, only the string table (`.strtab`) and the symbol tables that point into it
  differ; every code section is equal.
* `ftokc.so` is freestanding (no libc, no NEEDED entry) and built with Ubuntu 24.04's gcc 13.3.0.
* `kfx.cu` includes `kf.cu` and `kfx_map.h`; `onormpv.cu` includes `onorm.cu`.

## How to check

`verify.sh` unpacks the kernel layers from the committed image (`crane`), rebuilds every binary from the sources here and
compares the sha256 with every layer's copy:

```
./verify.sh <path to nvcc 13.4>
```

It builds `onorm.cubin` and `onormpv.cubin` at `/opt/s1eng/s1eng` when that directory can be created, and `ftokc.so` inside
`ubuntu:24.04` (docker) for gcc 13.3.

## Authors

The engine is the team's work: rgh, ruj, kai and pjs (per binary in the table above).

## Licence

See `LICENSE` (MIT).
