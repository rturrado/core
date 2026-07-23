/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/Layout.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <random>

namespace mlir::qco {

Layout Layout::random(const size_t nqubits, const size_t seed) {
  SmallVector<size_t> hwIndices(nqubits);
  std::iota(hwIndices.begin(), hwIndices.end(), size_t{0});
  std::ranges::shuffle(hwIndices, std::mt19937_64{seed});

  Layout layout(nqubits);
  for (size_t prog = 0; prog < nqubits; ++prog) {
    layout.add(prog, hwIndices[prog]);
  }
  return layout;
}

void Layout::add(const size_t prog, const size_t hw) {
  assert(prog < nqubits_ && "program index out of bounds");
  assert(hw < nqubits_ && "hardware index out of bounds");
  assert(!programToHardware_.contains(prog) && "program index already mapped");
  assert(!hardwareToProgram_.contains(hw) && "hardware index already mapped");
  programToHardware_[prog] = hw;
  hardwareToProgram_[hw] = prog;
}

size_t Layout::getProgramIndex(const size_t hw) const {
  const auto it = hardwareToProgram_.find(hw);
  assert(it != hardwareToProgram_.end() && "hardware index not mapped");
  return it->second;
}

size_t Layout::getHardwareIndex(const size_t prog) const {
  const auto it = programToHardware_.find(prog);
  assert(it != programToHardware_.end() && "program index not mapped");
  return it->second;
}

void Layout::swap(const size_t hwA, const size_t hwB) {
  const auto itA = hardwareToProgram_.find(hwA);
  const auto itB = hardwareToProgram_.find(hwB);
  assert(itA != hardwareToProgram_.end() && "hardware index not mapped");
  assert(itB != hardwareToProgram_.end() && "hardware index not mapped");
  const auto progA = itA->second;
  const auto progB = itB->second;
  itA->second = progB;
  itB->second = progA;
  programToHardware_[progA] = hwB;
  programToHardware_[progB] = hwA;
}

size_t Layout::nqubits() const { return nqubits_; }

} // namespace mlir::qco
