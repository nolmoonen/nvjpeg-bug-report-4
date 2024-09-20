#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>

#define CUDA_CALL(call)                                                   \
  do                                                                      \
  {                                                                       \
    cudaError_t res_ = call;                                              \
    if (cudaSuccess != res_)                                              \
    {                                                                     \
      std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                << " code=" << static_cast<unsigned int>(res_) << " ("    \
                << cudaGetErrorString(res_) << ") \"" << #call << "\"\n"; \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

#define NVJPEG_CALL(call)                                               \
  do                                                                    \
  {                                                                     \
    nvjpegStatus_t res_ = call;                                         \
    if (NVJPEG_STATUS_SUCCESS != res_)                                  \
    {                                                                   \
      std::cout << "nvJPEG error at " << __FILE__ << ":" << __LINE__    \
                << " code=" << static_cast<unsigned int>(res_) << "\n"; \
      std::exit(EXIT_FAILURE);                                          \
    }                                                                   \
  } while (0)

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cout << "usage: repro <image>\n";
    return EXIT_FAILURE;
  }

  std::ifstream input(argv[1], std::ios::in | std::ios::binary);
  if (!input.is_open())
  {
    std::cout << "failed to open " << argv[1];
  }
  input.seekg(0, std::ios::end);
  const std::streampos size = input.tellg();
  std::vector<uint8_t> buffer(size);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(buffer.data()), size);
  input.close();

  cudaStream_t stream = nullptr;

  nvjpegHandle_t nv_handle;
  NVJPEG_CALL(nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, nullptr, nullptr,
                             NVJPEG_FLAGS_REDUCED_MEMORY_DECODE, &nv_handle));

  int num_components;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  NVJPEG_CALL(nvjpegGetImageInfo(nv_handle, buffer.data(), buffer.size(),
                                 &num_components, &subsampling, widths,
                                 heights));
  std::cout << "number of pixels: "
            << (static_cast<size_t>(widths[0]) * heights[0]) << "\n";

  nvjpegJpegState_t jpeg_state{};
  NVJPEG_CALL(nvjpegJpegStateCreate(nv_handle, &jpeg_state));

  nvjpegJpegDecoder_t nvjpeg_decoder{};
  NVJPEG_CALL(nvjpegDecoderCreate(nv_handle, NVJPEG_BACKEND_GPU_HYBRID,
                                  &nvjpeg_decoder));
  nvjpegJpegState_t nvjpeg_decoupled_state;
  NVJPEG_CALL(nvjpegDecoderStateCreate(nv_handle, nvjpeg_decoder,
                                       &nvjpeg_decoupled_state));

  nvjpegBufferPinned_t pinned_buffer;
  NVJPEG_CALL(nvjpegBufferPinnedCreate(nv_handle, nullptr, &pinned_buffer));
  nvjpegBufferDevice_t device_buffer;
  NVJPEG_CALL(nvjpegBufferDeviceCreate(nv_handle, nullptr, &device_buffer));

  nvjpegJpegStream_t jpeg_stream;
  NVJPEG_CALL(nvjpegJpegStreamCreate(nv_handle, &jpeg_stream));

  nvjpegDecodeParams_t nvjpeg_decode_params;
  NVJPEG_CALL(nvjpegDecodeParamsCreate(nv_handle, &nvjpeg_decode_params));

  NVJPEG_CALL(
      nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, pinned_buffer));
  NVJPEG_CALL(
      nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
  NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params,
                                                NVJPEG_OUTPUT_UNCHANGED));

  NVJPEG_CALL(nvjpegJpegStreamParse(nv_handle, buffer.data(), buffer.size(), 0,
                                    0, jpeg_stream));

  NVJPEG_CALL(nvjpegDecodeJpegHost(nv_handle, nvjpeg_decoder,
                                   nvjpeg_decoupled_state, nvjpeg_decode_params,
                                   jpeg_stream));

  NVJPEG_CALL(nvjpegDecodeJpegTransferToDevice(
      nv_handle, nvjpeg_decoder, nvjpeg_decoupled_state, jpeg_stream, stream));

  nvjpegImage_t img;
  for (int c = 0; c < num_components; ++c)
  {
    CUDA_CALL(cudaMalloc(&(img.channel[c]), widths[c] * heights[c]));
    img.pitch[c] = widths[c];
  }

  NVJPEG_CALL(nvjpegDecodeJpegDevice(nv_handle, nvjpeg_decoder,
                                     nvjpeg_decoupled_state, &img, stream));

  CUDA_CALL(cudaStreamSynchronize(stream));

  NVJPEG_CALL(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
  NVJPEG_CALL(nvjpegJpegStreamDestroy(jpeg_stream));
  NVJPEG_CALL(nvjpegBufferPinnedDestroy(pinned_buffer));
  NVJPEG_CALL(nvjpegBufferDeviceDestroy(device_buffer));
  NVJPEG_CALL(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));
  NVJPEG_CALL(nvjpegDecoderDestroy(nvjpeg_decoder));
  NVJPEG_CALL(nvjpegJpegStateDestroy(jpeg_state));

  for (int c = 0; c < num_components; ++c)
  {
    CUDA_CALL(cudaFree(img.channel[c]));
  }
}
