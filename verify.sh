#!/bin/sh
# Rebuild UID 220's compiled kernels from the sources in this folder and compare them with the files in the committed
# image (every layer's copy of each binary is checked: final copies in layers 20 / 21 / 22, earlier ones in 16 / 17 / 19).
# usage: ./verify.sh <nvcc 13.4>      needs: crane, tar, sha256sum, docker (for ftokc.so: ubuntu:24.04 gcc 13.3)
set -e
NVCC134=${1:?path to nvcc 13.4}
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=ghcr.io/johnkell9801-hub/s1q9
REF=$IMG@sha256:f1422b06f6098ad7b8939c1e05a48b82493d2ccc672912b92e21491256ca0898
W=$(mktemp -d)
FAIL=0
layer() {  # layer <n> <digest>: unpack one layer's /opt/s1eng/s1eng into $W/L<n>
  mkdir -p "$W/L$1"; crane blob "$IMG@$2" | tar -xz -C "$W/L$1" opt/s1eng/s1eng 2>/dev/null || true
}
layer 16 sha256:184d097c857b374eb0f6d175159d810178ec754667d6f73b871130574a021970
layer 17 sha256:033c5dc751ab17bd19decf5ef83f6ef36d431a0adb8862aebc120390e80a30f2
layer 19 sha256:acb61750b3ebe6bbf9916f67c7a1d1798277e3fd5b5f4a3e4a578ba7f86e82c2
layer 20 sha256:b2d1b2849e8291a794267ed29e97bdf856661f7c5191d7de194fb4a09b336f72
layer 21 sha256:849ddd5f796b4e0ef3282a1e0186f33dc8ee8d3fa105281f8c7618fcb6ca5a72
layer 22 sha256:77733d5949dc2d7730f890465b8d816db3c81e43f77053c32a4a491d3de6c7ef
mkdir -p "$W/b" && cd "$W/b" && cp "$HERE"/*.cu "$HERE"/*.c "$HERE"/*.h .
build() {  # build <nvcc> <log> <args...>
  nv=$1; log=$2; shift 2
  if ! "$nv" "$@" > "$log" 2>&1; then echo "build failed: $nv $*"; tail -20 "$log"; exit 1; fi
}
build_inplace() {  # build_inplace <nvcc> <binary> <args>: byte-identical only from /opt/s1eng/s1eng (path CRC in a symbol)
  if mkdir -p /opt/s1eng/s1eng 2>/dev/null && [ -w /opt/s1eng/s1eng ] && [ ! -e /opt/s1eng/s1eng/onorm.cu ]; then
    cp "$HERE"/*.cu "$HERE"/*.h /opt/s1eng/s1eng/ && (cd /opt/s1eng/s1eng && "$1" $3 > /dev/null 2>&1) &&
      cp "/opt/s1eng/s1eng/$2" "$W/b/$2" && rm -f /opt/s1eng/s1eng/*.cu /opt/s1eng/s1eng/*.h "/opt/s1eng/s1eng/$2"
  else
    echo "($2: /opt/s1eng/s1eng not usable here; built in place, only .strtab / symbol tables may differ)"
    "$1" $3 > "$2.log" 2>&1
  fi
}
build_gcc() {  # ftokc.so with Ubuntu 24.04's gcc 13.3
  docker run --rm -v "$W/b:/w" -w /w ubuntu:24.04 sh -c "apt-get update -qq >/dev/null && apt-get install -y -qq gcc >/dev/null 2>&1 && $1"
}
check() {  # check <name>: the rebuilt file against every layer that has it
  a=$(sha256sum "$1" | cut -d' ' -f1)
  for n in 22 21 20 19 17 16; do
    f="$W/L$n/opt/s1eng/s1eng/$1"; [ -f "$f" ] || continue
    b=$(sha256sum "$f" | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then echo "layer $n $1: byte-identical ($a)"; else echo "layer $n $1: DIFFERS (rebuilt $a, image $b)"; FAIL=1; fi
  done
}
build "$NVCC134" ngemm5.cubin.log -arch=sm_120a -O3 -cubin -o ngemm5.cubin ngemm5.cu
build "$NVCC134" ngemm9b.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9b.cubin ngemm9b.cu
build "$NVCC134" ngemm9b_wn.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngemm9b_wn.cubin ngemm9b.cu
build "$NVCC134" ngemm9b_wnt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngemm9b_wnt.cubin ngemm9b.cu
build "$NVCC134" ngemm9t.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9t.cubin ngemm9t.cu
build "$NVCC134" ngemm9t_wn.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngemm9t_wn.cubin ngemm9t.cu
build "$NVCC134" ngemm9t_wnt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngemm9t_wnt.cubin ngemm9t.cu
build "$NVCC134" ngemm9l.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9l.cubin ngemm9l.cu
build "$NVCC134" ngr.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngr.cubin ngr.cu
build "$NVCC134" ngr_wn.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngr_wn.cubin ngr.cu
build "$NVCC134" xa.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o xa.cubin xa.cu
build "$NVCC134" kkt2n.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o kkt2n.cubin kkt2n.cu
build "$NVCC134" ffs.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ffs.cubin ffs.cu
build "$NVCC134" pron.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DPRON_SHIP -o pron.cubin pron.cu
build_inplace "$NVCC134" onorm.cubin '-O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onorm.cubin onorm.cu'
build "$NVCC134" kf.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o kf.cubin kf.cu
build "$NVCC134" kf_wn.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DKF_WTL -DKF_NMAJ -o kf_wn.cubin kf.cu
build "$NVCC134" kf_wnt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DKF_WTL -DKF_NMAJ -DTROW -o kf_wnt.cubin kf.cu
build "$NVCC134" skx6.cubin.log -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6.cubin skx6.cu
build "$NVCC134" skx6t.cubin.log -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6t.cubin skx6t.cu
build "$NVCC134" rn.cubin.log -cubin -arch=sm_120a -O3 -fmad=false -Xptxas -v -o rn.cubin rn.cu
build "$NVCC134" ngemm7.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -o ngemm7.cubin ngemm7.cu
build "$NVCC134" kfx.cubin.log -arch=sm_120a -O3 -cubin -Xptxas -v -o kfx.cubin kfx.cu
build "$NVCC134" zcin.cubin.log -cubin -arch=sm_120a -O3 -Xptxas -v -o zcin.cubin zcin.cu
build "$NVCC134" xa3.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o xa3.cubin xa3.cu
build_inplace "$NVCC134" onormpv.cubin '-O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onormpv.cubin onormpv.cu'
build "$NVCC134" ngemm9b_trow.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngemm9b_trow.cubin ngemm9b.cu
build "$NVCC134" ngemm9t_trow.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngemm9t_trow.cubin ngemm9t.cu
build "$NVCC134" ngemm7_trow.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -DTROW -o ngemm7_trow.cubin ngemm7.cu
build "$NVCC134" ngr_trow.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngr_trow.cubin ngr.cu
build "$NVCC134" ngr_wnt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngr_wnt.cubin ngr.cu
build "$NVCC134" kf_trow.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o kf_trow.cubin kf.cu
build "$NVCC134" kfx_trow.cubin.log -arch=sm_120a -O3 -cubin -Xptxas -v -DTROW -o kfx_trow.cubin kfx.cu
build "$NVCC134" ngemm8.cubin.log -O3 -arch=sm_120a -cubin -o ngemm8.cubin ngemm8.cu
build "$NVCC134" ngemm7n.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -o ngemm7n.cubin ngemm7n.cu
build "$NVCC134" ngemm7w.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -DNG7_WTL -o ngemm7w.cubin ngemm7w.cu
build "$NVCC134" ngemm7wn.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -DNG7_WTL -o ngemm7wn.cubin ngemm7wn.cu
build "$NVCC134" q8.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8.cubin q8.cu
build "$NVCC134" q8k.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8k.cubin q8k.cu
build "$NVCC134" q8wt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8wt.cubin q8wt.cu
build "$NVCC134" q8v.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8v.cubin q8v.cu
build "$NVCC134" q8pb.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8pb.cubin q8pb.cu
build_gcc 'gcc -O2 -fPIC -shared -nostdlib -ffreestanding -fno-builtin -o ftokc.so ftokc.c'
check ngemm5.cubin
check ngemm9b.cubin
check ngemm9b_wn.cubin
check ngemm9b_wnt.cubin
check ngemm9t.cubin
check ngemm9t_wn.cubin
check ngemm9t_wnt.cubin
check ngemm9l.cubin
check ngr.cubin
check ngr_wn.cubin
check xa.cubin
check kkt2n.cubin
check ffs.cubin
check pron.cubin
check onorm.cubin
check kf.cubin
check kf_wn.cubin
check kf_wnt.cubin
check skx6.cubin
check skx6t.cubin
check rn.cubin
check ngemm7.cubin
check kfx.cubin
check zcin.cubin
check xa3.cubin
check onormpv.cubin
check ngemm9b_trow.cubin
check ngemm9t_trow.cubin
check ngemm7_trow.cubin
check ngr_trow.cubin
check ngr_wnt.cubin
check kf_trow.cubin
check kfx_trow.cubin
check ngemm8.cubin
check ngemm7n.cubin
check ngemm7w.cubin
check ngemm7wn.cubin
check q8.cubin
check q8k.cubin
check q8wt.cubin
check q8v.cubin
check q8pb.cubin
check ftokc.so
echo "image: $REF"
rm -rf "$W"
exit $FAIL
