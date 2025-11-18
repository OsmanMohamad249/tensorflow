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

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_dimensions.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using PjRtDeviceDimensionsAndInt = std::pair<PjRtDeviceDimensions, int32_t>;

// Helper to get a TPU topology description.
absl::StatusOr<std::unique_ptr<PjRtTopologyDescription>> GetTpuTopology() {
  return GetCApiTopology("tpu", "TPU v2:4x4");
}

TEST(PjRtCApiTopologyDescriptionTpuTest, IsSubsliceTopology) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  // The default TPU topology is not a subslice.
  EXPECT_THAT(topology->is_subslice_topology(), false);
}

TEST(PjRtCApiTopologyDescriptionTpuTest, SubsliceTopology) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtTopologyDescription> topology,
                          GetTpuTopology());
  PjRtDeviceDimensions chips_per_host_bounds = {2, 2, 1};
  PjRtDeviceDimensions host_bounds = {1, 1, 1};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtTopologyDescription> subslice_topology,
      topology->Subslice(chips_per_host_bounds, host_bounds));
  EXPECT_THAT(subslice_topology->is_subslice_topology(), true);
  EXPECT_THAT(subslice_topology->DeviceDescriptions().size(), 8);
}

}  // namespace
}  // namespace xla
