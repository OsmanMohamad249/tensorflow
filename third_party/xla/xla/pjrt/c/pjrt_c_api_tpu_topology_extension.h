/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_C_PJRT_C_API_TPU_TOPOLOGY_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_TPU_TOPOLOGY_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality related to TPU topology.

#define PJRT_API_TPU_TOPOLOGY_EXTENSION_VERSION 1

struct PJRT_TpuTopology_Subslice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_TopologyDescription* topology;
  const int32_t* chips_per_host_bounds;
  size_t chips_per_host_bounds_num_dims;
  const int32_t* host_bounds;
  size_t host_bounds_num_dims;
  PJRT_TopologyDescription* subslice_topology;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_Subslice_Args, subslice_topology);

// Returns a subslice topology of the given topology.
typedef PJRT_Error* PJRT_TpuTopology_Subslice(
    PJRT_TpuTopology_Subslice_Args* args);

struct PJRT_TpuTopology_IsSubsliceTopology_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_TopologyDescription* topology;
  bool is_subslice_topology;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_IsSubsliceTopology_Args,
                          is_subslice_topology);

// Returns true if the topology is a subslice topology.
typedef PJRT_Error* PJRT_TpuTopology_IsSubsliceTopology(
    PJRT_TpuTopology_IsSubsliceTopology_Args* args);

typedef struct PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* client_topology;
  const PJRT_TopologyDescription* subslice_topology;
  const int32_t* subslice_origin;
  size_t subslice_origin_dim_num;
  int32_t full_device_id;

  int32_t subslice_device_id;  // out
} PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args;
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args, subslice_device_id);

// Returns the subslice device id for the given full device id.
typedef PJRT_Error* PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId(
    PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId_Args* args);

typedef struct PJRT_TpuTopology_ReplaceHostBounds_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  const int32_t* host_bounds;
  size_t host_bounds_dim_num;

  PJRT_TopologyDescription* new_topology;  // out
} PJRT_TpuTopology_ReplaceHostBounds_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_ReplaceHostBounds_Args,
                          new_topology);

// Returns a new PjRtTopologyDescription by replacing the host bounds of the
// input `topology` with the provided `host_bounds`.
typedef PJRT_Error* PJRT_TpuTopology_ReplaceHostBounds(
    PJRT_TpuTopology_ReplaceHostBounds_Args* args);

typedef struct PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  bool is_enhanced_barrier_enabled;  // out
} PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args,
                          is_enhanced_barrier_enabled);

// Returns true if the enhanced barrier is enabled in the given TPU topology.
typedef PJRT_Error* PJRT_TpuTopology_IsEnhancedBarrierEnabled(
    PJRT_TpuTopology_IsEnhancedBarrierEnabled_Args* args);

typedef struct PJRT_TpuTopology_HasLimitedIciConnectivity_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  bool has_limited_ici_connectivity;  // out
} PJRT_TpuTopology_HasLimitedIciConnectivity_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_HasLimitedIciConnectivity_Args,
                          has_limited_ici_connectivity);

// Returns true if the given TPU topology has limited ICI connectivity.
typedef PJRT_Error* PJRT_TpuTopology_HasLimitedIciConnectivity(
    PJRT_TpuTopology_HasLimitedIciConnectivity_Args* args);

typedef struct PJRT_TpuTopology_IsReachableOverLimitedIci_Args {
  size_t struct_size;

  const PJRT_TopologyDescription* topology;
  int32_t source_chip_id;
  int32_t dest_chip_id;
  bool is_reachable_over_limited_ici;  // out
} PJRT_TpuTopology_IsReachableOverLimitedIci_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_IsReachableOverLimitedIci_Args,
                          is_reachable_over_limited_ici);

// Returns true if `source_chip_id` can directly reach `dest_chip_id` on a TPU
// topology with limited ICI routing.
typedef PJRT_Error* PJRT_TpuTopology_IsReachableOverLimitedIci(
    PJRT_TpuTopology_IsReachableOverLimitedIci_Args* args);

typedef struct PJRT_TpuTopology_Extension {
  PJRT_Extension_Base base;
  PJRT_TpuTopology_Subslice* subslice;
  PJRT_TpuTopology_IsSubsliceTopology* is_subslice_topology;
  PJRT_TpuTopology_SubsliceDeviceIdFromFullDeviceId*
      subslice_device_id_from_full_device_id;
  PJRT_TpuTopology_ReplaceHostBounds* replace_host_bounds;
  PJRT_TpuTopology_IsEnhancedBarrierEnabled* is_enhanced_barrier_enabled;
  PJRT_TpuTopology_HasLimitedIciConnectivity* has_limited_ici_connectivity;
  PJRT_TpuTopology_IsReachableOverLimitedIci* is_reachable_over_limited_ici;
} PJRT_TpuTopology_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TpuTopology_Extension,
                          is_reachable_over_limited_ici);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_TPU_TOPOLOGY_EXTENSION_H_
