#include "ck_tile/host.hpp"
#include "vector_add.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "10240", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = DataType;

    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    // Step 1: Allocate Memory on Host and Device
    // Step 1.1 Allocate host memory for A, B and fill random values
    ck_tile::HostTensor<XDataType> x_host_a({m, n});
    ck_tile::HostTensor<XDataType> x_host_b({m, n});

    ck_tile::HostTensor<YDataType> y_host_ref({m, n});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n});

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_b);

    // Step 1.2 Allocate device memory and copy data from host
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());

    // Step 2: Problem Setting
    // Step 2.1 Set the following variables
    using Vector     = ck_tile::sequence<1, 4>;
    using BlockWarps = ck_tile::sequence<1, 8>;
    // Step 2.2 Derive the following variables
    using BlockTile  = ck_tile::sequence<1, 2048>;
    using WarpTile   = ck_tile::sequence<1, 256>;

    // Step 2.3 Define "Shape" for "Problem", and define "Problem" for "Kernel"
    using Shape = ck_tile::VectorAddShape<BlockWarps, BlockTile, WarpTile, Vector>;
    using Problem = ck_tile::VectorAddProblem<XDataType, ComputeDataType, YDataType, Shape>;

    // Step 5 Launch Kernel
    constexpr ck_tile::index_t kBlockSize  = 512; // This value is related with above Problem Setting
    constexpr ck_tile::index_t kBlockPerCu = 1;
    ck_tile::index_t kGridSize             = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "block x-size = " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "grid size " << kGridSize << std::endl;

    using Kernel = ck_tile::VectorAdd<Problem>;
    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                                       static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()),
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       m,
                                       n));

    std::size_t num_btype = sizeof(XDataType) * m * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        ck_tile::reference_vectoradd<XDataType, YDataType>(
           x_host_a, x_host_b, y_host_ref);
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, y_host_ref);

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
}
