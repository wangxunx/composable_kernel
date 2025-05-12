#pragma once

#include "block_gemm_asmem_bsmem_creg.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

namespace ck_tile {

// Default policy for BlockGemmPipelineAGmemBGmemCReg
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmPipelineAGmemBGmemCRegDefaultPolicy
{
    // Step 3.1: GEMM Tile Distribution Policy
    // DRAM -> register
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;      // ck::half_t
        constexpr index_t kBlockSize = Problem::kBlockSize;                 // kBLockSize = 256

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;         // kMPerBlock = 256
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;         // kKPerBlock = 32

        constexpr index_t K1 = 16 / sizeof(ADataType);                      // K1 = 8, vector load size 8, vector_K
        constexpr index_t K0 = kKPerBlock / K1;                             // K0 = 4, thread per Warp K

        constexpr index_t M2 = get_warp_size() / K0;                        // M2 = 16, thread per Warp M
        constexpr index_t M1 = kBlockSize / get_warp_size();                // M1 = 4, warpPerBlock M
        constexpr index_t M0 = kMPerBlock / (M2 * M1);                      // M0 = 4, repeat M

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,                         // Replication shapes
                                       tuple<sequence<M0, M1, M2>,          // Hierarchy shapes
                                             sequence<K0, K1>>,  
                                       tuple<sequence<1>, sequence<1, 2>>,  // Parallel shapes indexing
                                       tuple<sequence<1>, sequence<2, 0>>,  // Parallel shapes indexing
                                       sequence<1, 2>,                      // Yield shapes indexing: repeat & vector
                                       sequence<0, 1>>{});                  // Yield shapes indexing: repeat & vector
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;      // ck::half_t
        constexpr index_t kBlockSize = Problem::kBlockSize;                 // kBlockSize = 256

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;         // kNPerBlock = 128
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;         // kKPerBlock = 32

        constexpr index_t K1 = 16 / sizeof(BDataType);                      // K1 = 8, vector load size 8, vector_K
        constexpr index_t K0 = kKPerBlock / K1;                             // K0 = 4, thread per Warp K

        constexpr index_t N2 = get_warp_size() / K0;                        // N2 = 16, thread per warp N
        constexpr index_t N1 = kBlockSize / get_warp_size();                // N1 = 4, warpPerBlock N
        constexpr index_t N0 = kNPerBlock / (N2 * N1);                      // N0 = 2, repeat N

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,                         // Replication shapes
                                       tuple<sequence<N0, N1, N2>,          // Hierarchy shapes
                                             sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,  // Parallel shapes indexing
                                       tuple<sequence<1>, sequence<2, 0>>,  // Parallel shapes indexing
                                       sequence<1, 2>,                      // Yield shapes indexing: repeat & vector
                                       sequence<0, 1>>{});                  // Yield shapes indexing: repeat & vector
    }

    // Register -> LDS
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        const index_t kMPerBlock = Problem::BlockGemmShape::kM;
        const index_t kKPerBlock = Problem::BlockGemmShape::kK;
        const index_t kKPack     = 8;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                    make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        const index_t kNPerBlock = Problem::BlockGemmShape::kN;
        const index_t kKPerBlock = Problem::BlockGemmShape::kK;
        const index_t kKPack     = 8;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                    make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        
        return b_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return BlockGemmASmemBSmemCReg<Problem>{};
    }
};

}
