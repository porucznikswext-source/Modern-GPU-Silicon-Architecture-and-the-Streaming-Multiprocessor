# Modern-GPU-Silicon-Architecture-and-the-Streaming-Multiprocessor

```

Modern GPU silicon is engineered around massive data-parallel throughput rather than low-latency
single-thread execution. Where modern x86-64 CPUs allocate substantial die area to out-of-order
execution logic, deep branch prediction trees, and massive multi-level cache hierarchies, the Graphics
Processing Unit (GPU) commits the overwhelming majority of its transistor budget directly to arithmetic
logic units (ALUs), large register files, and high-bandwidth memory interfaces.
Achieving mechanical sympathy on NVIDIA architectures-from Ampere and Ada Lovelace to Hopper and
Blackwell-requires an exact understanding of the physical layout of the silicon, the internal execution
pipelines of the Streaming Multiprocessor (SM), and the memory interconnects that feed these compute
engines.

+-----------------------------------------------------------------------+
|                       GPU Silicon Floorplan                           |
| +-------------------------------------------------------------------+ |
| |                  GPC (Graphics Processing Cluster)                | |
| | +-------------------+ +-------------------+ +-------------------+ | |
| | |       TPC 0       | |       TPC 1       | |       TPC N       | | |
| | | +---------------+ | | +---------------+ | | +---------------+ | | |
| | | |  SM 0 |  SM 1 | | | |  SM 2 |  SM 3 | | | |  SM ... | SM k  | | | 
| | | +---------------+ | | +---------------+ | | +---------------+ | | |
| | +-------------------+ +-------------------+ +-------------------+ | |
| +-------------------------------------------------------------------+ |
| +-------------------------------------------------------------------+ |
| |               High-Bandwidth Crossbar / Network-on-Chip           | |
| +-------------------------------------------------------------------+ |
| |                   L2 Cache Slices (Shared Subsystem)              | |
| +-------------------------------------------------------------------+ |
| |              Memory Controllers (HBM3e / GDDR6X PHYs)             | |
| +-------------------------------------------------------------------+ |

+-----------------------------------------------------------------------+

Silicon Floorplan: GPCs, TPCs, and Memory Subsystems
A modern high-performance GPU die is organized hierarchically into scalable hardware blocks:

1. GPC (Graphics Processing Cluster): The primary macro-unit of the GPU die. A high-end GPU contains
multiple GPCs (e.g., 8 in GA100, 8 in GH100). Each GPC functions as a self-contained compute engine
with dedicated rasterization pipelines (if graphics are enabled) and multiple Texture Processing Clusters.

2. TPC (Texture Processing Cluster): Each GPC contains multiple TPCs. A TPC groups two (or more,
depending on microarchitecture) Streaming Multiprocessors along with polymorph engines and texture
mapping units (TMUs).

3. High-Speed Interconnect (Crossbar / Network-on-Chip): Connects all GPCs to the unified Level-2 (L2)
cache slices, memory controllers, NVLink high-speed interfaces, and the PCIe host interface.

4. L2 Cache Slices & Memory Controllers: The L2 cache is physically partitioned across multiple memory
controllers. Each partition services a fraction of the physical address space, interfacing directly with High
Bandwidth Memory (HBM3/HBM3e) stacks or GDDR6X channels via high-speed physical layer interfaces (PHYs).


The Anatomy of the Streaming Multiprocessor (SM)
The Streaming Multiprocessor is the core computational engine of the NVIDIA GPU. While architectural
revisions introduce specialized hardware (such as Transformer Engines or Asynchronous Copy units), the
core internal structure of modern SMs follows a quad-partitioned topology.

+-----------------------------------------------------------------------+
|                    Streaming Multiprocessor (SM)                      |
| +-------------------------------------------------------------------+ |
| |                      Unified L1 Cache / Shared Memory               |
| +-------------------------------------------------------------------+ |
| |  Sub-Core 0         Sub-Core 1         Sub-Core 2    Sub-Core 3   | |
| |  +---------------+  +---------------+  +-----------+ +----------+ | |
| |  | Warp Sched.   |  | Warp Sched.   |  | Warp Sch. | | Warp Sch.| | |
| |  | Dispatch Unit |  | Dispatch Unit |  | Dispatch  | | Dispatch | | |
| |  | Reg File (16K)|  | Reg File (16K)|  | Reg File  | | Reg File | | |
| |  | 16x FP32 Core |  | 16x FP32 Core |  | 16x FP32  | | 16x FP32 | | |
| |  | 16x INT32 Core|  | 16x INT32 Core|  | 16x INT32 | | 16x INT32| | |
| |  | 8x  FP64 Core |  | 8x  FP64 Core |  | 8x  FP64  | | 8x  FP64 | | |
| |  | 4x  SFU       |  | 4x  SFU       |  | 4x  SFU   | | 4x  SFU  | | |
| |  | 1x Tensor Core|  | 1x Tensor Core|  | 1x Tensor | | 1x Tensor| | |
| |  | 8x LSU        |  | 8x LSU        |  | 8x LSU    | | 8x LSU   | | |
| |  +---------------+  +---------------+  +-----------+ +----------+ | |
| +-------------------------------------------------------------------+ |
+-----------------------------------------------------------------------+

An SM is split into four distinct sub-cores (often referred to as SM processing blocks or partitions). Each
sub-core operates independently and contains its own dedicated:
• Instruction Cache (I-Cache) & Buffer: Stores and prefetches instructions for resident warps.
• Warp Scheduler & Dispatch Unit: Identifies warps eligible for execution (whose operands are ready and
execution units are free) and issues instructions each clock cycle.
• Register File Slice: Modern architectures typically allocate a total of 64K 32-bit registers (256 KB) per SM,
split evenly across the 4 partitions (16K 32-bit registers per sub-core).
• Execution Units:
• FP32 Cores: Single-precision IEEE-754 floating-point arithmetic.
• INT32 Cores: 32-bit integer arithmetic and pointer address generation. Modern architectures support
concurrent FP32 + INT32 execution.
• FP64 Cores: Double-precision floating-point execution (densely populated in datacenter chips like
A100/H100; sparse in client chips like RTX 4090).
• Special Function Units (SFUs): Perform transcendental functions (sin, cos, exp, log), square roots, and
reciprocal approximations.
• Load/Store Units (LSUs): Calculate source/destination addresses and interface with the L1 cache.
• Tensor Cores: Specialized Mixed-Precision Matrix-Multiply-Accumulate (MMA) hardware executing dense
matrix operations directly at the hardware instruction level.

Shared Memory and L1 Cache Architecture
Modern SMs utilize a unified SRAM array that serves both as the L1 Data Cache and software-managed
Shared Memory.

Physical Unified SRAM Pool (e.g., 256 KB per SM on Hopper GH100)

+--------------------------------------------------------------------+
| Software-Managed Shared Memory (Static/Dynamic) | Hardware L1 Data |
| (Configurable: 0 KB to 228 KB)                  | (Remaining Pool) |
+--------------------------------------------------------------------+

The split between L1 Cache and Shared Memory is statically configured or dynamically carved out per
kernel launch via cudaFuncSetAttribute(). Shared memory access occurs over 32 independent banks
(each 4 bytes wide). If threads within a warp access distinct banks, the SRAM array delivers full bandwidth
in a single clock cycle. If multiple threads target different 4-byte words within the same bank, the hardware
serializes the transaction, causing bank conflicts.

Hardware Metrics & Theoretical Throughput Math
To accurately benchmark and profile custom kernels, you must calculate the theoretical ceilings of the
targeted hardware.

# 1. Peak Compute Throughput (FLOPS)
$$\text{Peak FLOPS} = N_{\text{SM}} \times N_{\text{Cores/SM}} \times f_{\text{boost}} \times
\text{Ops/Cycle}$$
Where:
• $N_{\text{SM}}$ is the active SM count.
• $N_{\text{Cores/SM}}$ is the number of functional units of the target type per SM.
• $f_{\text{boost}}$ is the sustained core clock frequency in Hz.
• $\text{Ops/Cycle} = 2$ for Fused Multiply-Add (FMA), which executes 1 multiply and 1 add in a single
cycle.

# 2. Peak Memory Bandwidth (Bytes/s)
$$\text{Peak Bandwidth} = \text{Bus Width (bits)} \times \text{Data Rate (Hz)} \times \frac{1}{8\text{
bits/byte}}$$

For HBM architectures (e.g., H100 SXM5 with a 5120-bit interface at 3.35 Gbps per pin):
$$\text{Bandwidth} = \frac{5120 \times 3.35 \times 10^9}{8} \approx 2.144 \times 10^{12} \text{ B/s} = 2.144 \text{ TB/s}$$

Programmatic Hardware Extraction
The following production-ready C++ diagnostic harness uses the CUDA Runtime API to extract internal
topology metrics, compute configuration properties, and calculate theoretical throughput directly from the
silicon:

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__  \
                      << " -> " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
// Returns FP32 ALUs per SM based on Compute Capability Major.Minor
inline int getFP32CoresPerSM(int major, int minor) {
    switch ((major << 4) + minor) {
        case 0x70: case 0x72: return 64;   // Volta
        case 0x75:            return 64;   // Turing
        case 0x80:            return 64;   // Ampere (GA100)
        case 0x86: case 0x87: return 128;  // Ampere (GA102/GA104/GA106)
        case 0x89:            return 128;  // Ada Lovelace
        case 0x90:            return 128;  // Hopper
        default:              return 128;  // Default fallback
    }
}
int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices detected.\n";
        return 0;
    }
    for (int dev = 0; dev < deviceCount; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::cout << "====================================================\n";
        std::cout << "DEVICE ID " << dev << ": " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." 
                  << prop.minor << "\n";
        std::cout << "====================================================\n";
        // SM Topography
        int coresPerSM = getFP32CoresPerSM(prop.major, prop.minor)

 int totalCores = prop.multiProcessorCount * coresPerSM;
        double clockGHz = prop.clockRate * 1e-6; // kHz to GHz
        std::cout << "[Compute Topology]\n";
        std::cout << "  Streaming Multiprocessors (SMs): " 
                  << prop.multiProcessorCount << "\n";
        std::cout << "  FP32 Cores per SM:               " 
                  << coresPerSM << "\n";
        std::cout << "  Total Active FP32 Cores:         " 
                  << totalCores << "\n";
        std::cout << "  Base/Boost Clock Rate:           " 
                  << std::fixed << std::setprecision(3) << clockGHz 
                  << " GHz\n";
        // Compute Theoretical Peak (FMA = 2 ops/cycle)
        double peakTFlops = (totalCores * (clockGHz * 1e9) * 2.0) / 1e12;
        std::cout << "  Theoretical Peak FP32:           " 
                  << std::setprecision(2) << peakTFlops << " TFLOPS\n\n";
        // Memory Subsystem
        double memClockGHz = prop.memoryClockRate * 1e-6;
        int busWidthBits = prop.memoryBusWidth;
        double peakBandwidthGBps = 
            (2.0 * prop.memoryClockRate * (busWidthBits / 8.0)) / 1e6;
        std::cout << "[Memory Hierarchy]\n";
        std::cout << "  Global Memory:                   " 
                  << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "  L2 Cache Size:                   " 
                  << (prop.l2CacheSize / 1024) << " KB\n";
        std::cout << "  Memory Bus Width:                " 
                  << busWidthBits << " bits\n";
        std::cout << "  Memory Clock Rate:               " 
                  << memClockGHz << " GHz\n";
        std::cout << "  Theoretical Peak Bandwidth:      " 
                  << peakBandwidthGBps << " GB/s\n\n";
        // SM Resource Limits
        std::cout << "[SM Execution Limits]\n";
        std::cout << "  Max Warps per SM:                " 
                  << (prop.maxThreadsPerMultiProcessor / 32) << "\n";
        std::cout << "  Max Threads per SM:              " 
                  << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Max Thread Blocks per SM:        " 
                  << prop.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  Registers per SM:                " 
                  << prop.regsPerMultiprocessor << " (32-bit)\n";
        std::cout << "  Shared Memory per SM (Max):      " 
                  << (prop.sharedMemPerMultiprocessor / 1024) << " KB\n";
        std::cout << "  Shared Memory per Block (Max):   " 
                  << (prop.sharedMemPerBlock / 1024) << " KB\n";
        std::cout << "====================================================\n\n";
    }
    return 0;
}

Mechanical Sympathy: Pipelining and Resource Balances
Writing performant CUDA kernels requires designing algorithms around the hard capacity constraints of
the SM. Every kernel launch incurs strict hardware resource tradeoffs:

+----------------------------------------------------------------------+
|                     Hardware Schedulability Matrix                   |
+------------------------------------+---------------------------------+
| Hardware Resource                  | Bottleneck Consequence          |
+------------------------------------+---------------------------------+
| Register Allocation (> 32 regs/th) | Drops resident active warps.    |
| Shared Memory Consumption (> 48KB) | Limits resident blocks per SM.  |
| Uncoalesced Global Access          | Amplifies memory controller PHY |
|                                    | requests, causing stall cycles. |
| Non-transposed Bank Access         | Serializes shared memory loads. |
+------------------------------------+---------------------------------+

1. Register Pressure: If a thread allocates 64 registers on an architecture with a 64K-register SM, the SM
cannot host more than $65,536 / 64 = 1,024$ concurrent threads (32 warps), limiting occupancy to 50% of
the hardware maximum (typically 64 warps on modern SMs).

2. Occupancy vs. Latency Hiding: The SM hides memory and pipeline latencies by rapidly
context-switching between active, resident warps in the register file without software overhead. When
hardware resources (registers, shared memory) are oversubscribed by a single block, fewer warps remain
available to the dispatch units to hide pipeline stalls.

```


