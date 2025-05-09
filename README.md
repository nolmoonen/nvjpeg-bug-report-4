# nvjpeg-bug-report-4

> [!CAUTION]
> Thanks to the nvJPEG team, this bug has been fixed in CUDA 12.9.

Demonstrates issues in `nvjpegDecodeJpegTransferToDevice` when decoding large images. The reproducer simply decodes an image.

Tested with CUDA 12.6.1 driver version 560.35.03 on Ubuntu 22.04.3.

## Example
Two grayscale images are provided, `test_46336.jpg` with has a width and a height of `46336` and `test_46344.jpg` has a width and a height of `46344` (one 8x8 block more horizontally and vertically). The smaller image has fewer pixels than `INT_MAX`, the larger image has more pixels than `INT_MAX`.

The smaller image is decoded succesfully, the larger image has issues. An example of the output of Compute Sanitizer for `./repro test_46344.jpg`:

```shell
========= Invalid __global__ write of size 2 bytes
=========     at void nvjpeg::DecodeSingleGPU::dcAcDecodeKernel<nvjpeg::DecodeSingleGPU::DcAcDecodeAdditionalArgs>(int, NppiSize, unsigned long, int, T1, short *, NppiSize, int, unsigned char, unsigned char)+0x8e0
=========     by thread (32,0,0) in block (66,724,0)
=========     Address 0x744ba9869200 is out of bounds
=========     and is 3,933,826,560 bytes before the nearest allocation at 0x744c94000000 of size 4,656,313,344 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x2dcdbf]
=========                in /lib/x86_64-linux-gnu/libcuda.so.1
=========     Host Frame:libcudart_static_4d8b33a106dceb3c07a56e26de61f2d53bb62a68 [0x19782]
=========                in /root/nvjpeg-bug-report-4/./build/repro
=========     Host Frame:cudaLaunchKernel [0x7d3ed]
=========                in /root/nvjpeg-bug-report-4/./build/repro
=========     Host Frame:libnvjpeg_static_6c51521374a636e9b3c487774fd9d242d6626083 [0x16b0fc]
=========                in /root/nvjpeg-bug-report-4/./build/repro
=========     Host Frame:libnvjpeg_static_a1e4dde9a2eb9fcc35810b86760c6c21c8b8e1d1 [0xdb946]
=========                in /root/nvjpeg-bug-report-4/./build/repro
=========     Host Frame:libnvjpeg_static_125692580b8a852592e92ea2462fa5f1592faf6d [0xdcb95]
=========                in /root/nvjpeg-bug-report-4/./build/repro
=========     Host Frame:nvjpegDecodeJpegTransferToDevice [0xa4617]
=========                in /root/nvjpeg-bug-report-4/./build/repro
=========     Host Frame:main [0x129e7]
=========                in /root/nvjpeg-bug-report-4/./build/repro
=========     Host Frame:__libc_start_call_main in ../sysdeps/nptl/libc_start_call_main.h:58 [0x29d8f]
=========                in /lib/x86_64-linux-gnu/libc.so.6
=========     Host Frame:__libc_start_main in ../csu/libc-start.c:392 [0x29e3f]
=========                in /lib/x86_64-linux-gnu/libc.so.6
=========     Host Frame:_start [0x11b64]
=========                in /root/nvjpeg-bug-report-4/./build/repro
```

An example of full Compute Sanitizer output on the first example image is in `out.txt`.
