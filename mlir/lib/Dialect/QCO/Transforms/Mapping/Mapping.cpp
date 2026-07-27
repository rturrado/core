/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/Graph.h"
#include "mlir/Dialect/QCO/Utils/Layout.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/PriorityQueue.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Threading.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "mapping-pass"

namespace mlir::qco {

using namespace mlir::qtensor;
using namespace mlir::utils;

#define GEN_PASS_DEF_MAPPINGPASS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

struct MappingPass : impl::MappingPassBase<MappingPass> {
private:
  using IndexPairType = std::pair<size_t, size_t>;
  using Window = SmallVector<IndexPairType>;
  using Wires = SmallVector<WireIterator>;
  using RecursiveRoutingStackItem = std::pair<Operation*, SmallVector<size_t>>;
  using RecursiveRoutingStack = SmallVector<RecursiveRoutingStackItem>;

  enum class RoutingMode : bool { Cold, Hot };

  /// Return the qubit values in `values`, preserving their relative order.
  static SmallVector<Value> getQubitValues(ValueRange values) {
    return to_vector(llvm::make_filter_range(
        values, [](Value value) { return isa<QubitType>(value.getType()); }));
  }

  class AugmentedDevice {
  public:
    explicit AugmentedDevice(
        const DenseSet<std::pair<size_t, size_t>>& couplingSet)
        : coupling_(couplingSet), dist_(coupling_.getDistMatrix()) {}

    /// Return the device's number of qubits.
    [[nodiscard]] size_t nqubits() const { return coupling_.getNumNodes(); }

    /// Return true if two qubits are adjacent.
    [[nodiscard]] bool areAdjacent(const size_t u, const size_t v) const {
      return dist_[u][v] == 1UL;
    }

    /// Return the length of the shortest path between two qubits.
    [[nodiscard]] size_t distanceBetween(const size_t u, const size_t v) const {
      const auto dist = dist_[u][v];
      if (dist == UINT64_MAX) {
        report_fatal_error("Failed to compute the distance between qubits " +
                           Twine(u) + " and " + Twine(v));
      }
      return dist;
    }

    /// Return the qubit identifiers.
    [[nodiscard]] SmallVector<size_t> qubits() const {
      return coupling_.getNodes();
    }

    /// Return all neighbours of a qubit.
    [[nodiscard]] ArrayRef<size_t> neighboursOf(const size_t u) const {
      return coupling_.getNeighbours(u);
    }

    /// Return the max degree (connectivity) of any qubit of the device.
    [[nodiscard]] size_t maxDegree() const { return coupling_.getMaxDegree(); }

  private:
    Graph coupling_;
    Graph::DistanceMatrix dist_;
  };

  /// Statistics collected while routing.
  struct Statistics {
    size_t nswaps{0};
  };

  /// Parameters influencing the behavior of the A* search algorithm.
  struct Parameters {
    float alpha;
    float lambda;
  };

  /// Utility-struct for routing functions.
  ///
  /// The wire index equals the hardware qubit index everywhere in the pass.
  /// `wires[hw]` is a `WireIterator` walking the SSA chain of hardware qubit
  /// `hw`. A hardware qubit that carries no program is stored as a sentinel
  /// `WireIterator` so that wire walkers skip that wire during a walk.
  /// `layout` is the single source of truth for the
  /// program-to-hardware mapping.
  struct RoutingBundle {
    Wires wires;
    Layout layout;
  };

  /// Describes a node in the A* search graph.
  struct Node {
    struct ComparePointer {
      bool operator()(const Node* lhs, const Node* rhs) const {
        return lhs->f > rhs->f;
      }
    };

    Layout layout;
    IndexPairType swap;
    Node* parent;
    size_t depth;
    float f;

    /// Construct a root node with the given layout. Initialize the
    /// sequence with an empty vector and set the cost to zero.
    explicit Node(Layout layout)
        : layout(std::move(layout)), parent(nullptr), depth(0), f(0) {}

    /// Construct a non-root node from its parent node. Apply the given swap to
    /// the layout of the parent node.
    Node(Node* parent, const IndexPairType& swap, const Window& window,
         const AugmentedDevice& device, const Parameters& params)
        : layout(parent->layout), swap(swap), parent(parent),
          depth(parent->depth + 1), f(0) {
      layout.swap(swap.first, swap.second);
      f = g(params.alpha) + h(window, device, params); // NOLINT
    }

    /// Return true, if the current SWAP sequence makes all gates in the front
    /// executable.
    [[nodiscard]] bool isGoal(const IndexPairType& front,
                              const AugmentedDevice& device) const {
      const auto [hw0, hw1] =
          layout.getHardwareIndices(front.first, front.second);
      return device.areAdjacent(hw0, hw1);
    }

  private:
    /// Calculate the path cost for the A* search algorithm.
    /// The path costs are the weighted sum of the currently required SWAPs.
    [[nodiscard]] float g(const float alpha) const {
      return alpha * static_cast<float>(depth);
    }

    /// Calculate the heuristic cost for the A* search algorithm.
    ///
    /// Computes the minimal number of SWAPs required to route each gate in
    /// each layer. For each gate, this is determined by the shortest distance
    /// between its hardware qubits. Intuitively, this is the number of SWAPs
    /// that a naive router would insert to route the layers (with a constant
    /// layout).
    [[nodiscard]] float h(const Window& window, const AugmentedDevice& device,
                          const Parameters& params) const {
      float costs{0};
      float decay{1.};

      for (const auto& [i, progs] : enumerate(window)) {
        const auto [prog0, prog1] = progs;
        const auto [hw0, hw1] = layout.getHardwareIndices(prog0, prog1);
        const size_t nswaps = device.distanceBetween(hw0, hw1) - 1;
        costs += decay * static_cast<float>(nswaps);
        decay *= params.lambda;
      }
      return costs;
    }
  };

  /// Describes the graph F of arXiv:1602.05150v3.
  struct FGraph {
    explicit FGraph(std::shared_ptr<AugmentedDevice> device)
        : f_(device->qubits()), device_(std::move(device)) {};

    /// Build F-graph: Add edges to F for each edge in the coupling graph.
    /// Note that this assumes that the coupling graph is directed, but
    /// symmetric (essentially: undirected).
    void construct(const Layout& from, const Layout& to) {
      for (const auto u : device_->qubits()) {
        for (const auto v : device_->neighboursOf(u)) {
          if (shouldAddEdge(u, v, from, to)) {
            f_.addEdge(u, v);
          }
        }
      }
    }

    /// Try to find a directed cycle in the F graph. If there is one,
    /// we can apply a happy swap chain. Note that this happy swap chain
    /// does not include the final back edge closing the cycle because the
    /// first SWAP changes the token (the qubit) on the target, invalidating
    /// the edge in F.
    [[nodiscard]] std::optional<SmallVector<IndexPairType>>
    findHappySWAPChain() const {
      const auto optCycle = f_.findCycle();
      if (!optCycle) {
        return std::nullopt;
      }
      const auto& cycle = *optCycle;

      SmallVector<IndexPairType> swaps;
      for (size_t i = cycle.size() - 1; i > 0; --i) {
        swaps.emplace_back(cycle[i], cycle[i - 1]);
      }
      return swaps;
    }

    /// Find an unhappy SWAP. That is, find an edge (u, v), where exchanging u
    /// and v, reduces u's distance to its target location (by one) and
    /// increases v's distance from 0 (already at the correct location) to one.
    [[nodiscard]] std::optional<IndexPairType> findUnhappySWAP() const {
      for (const auto u : f_.getNodes()) {
        for (const auto v : f_.getNeighbours(u)) {
          if (f_.getDegree(v) == 0) {
            return {{u, v}};
          }
        }
      }

      return std::nullopt;
    }

    /// Reset the F graph for rebuilding.
    void reset() { f_.clearEdges(); }

  private:
    /// Return true if moving the program qubit on hardware qubit `u` to
    /// hardware qubit `v` brings it closer to its destination hardware qubit.
    [[nodiscard]] bool shouldAddEdge(const size_t u, const size_t v,
                                     const Layout& from,
                                     const Layout& to) const {
      if (!from.hasProgramAt(u)) {
        return false;
      }
      const auto dest = to.getHardwareIndex(from.getProgramIndex(u));
      return device_->distanceBetween(v, dest) <
             device_->distanceBetween(u, dest);
    }

    Graph f_;
    std::shared_ptr<AugmentedDevice> device_;
  };

public:
  /// Construct default mapping pass.
  MappingPass() = default;

  /// Construct default mapping pass with options.
  explicit MappingPass(const MappingPassOptions& options)
      : MappingPassBase(options) {}

  /// Construct mapping from coupling set.
  explicit MappingPass(const DenseSet<std::pair<size_t, size_t>>& couplingSet,
                       const MappingPassOptions& options)
      : MappingPassBase(options),
        device(std::make_shared<AugmentedDevice>(couplingSet)) {}

protected:
  void runOnOperation() override {
    assert(alpha > 0 && "expected alpha > 0");
    assert(niterations > 0 && "expected niterations > 0");
    assert(ntrials > 0 && "expected ntrials > 0");

    if (!device) {
      llvm::reportFatalUsageError("No device specified!");
    }

    IRRewriter rewriter(&getContext());

    auto mod = getOperation();
    auto func = getEntryPoint(mod);
    if (!func) {
      mod.emitError() << "does not contain an entry point function";
      signalPassFailure();
      return;
    }

    auto comp = getComputation(func);
    if (failed(comp)) {
      signalPassFailure();
      return;
    }

    auto& body = func.getFunctionBody();
    const auto& extracts = *comp;

    if (extracts.size() > device->nqubits()) {
      func.emitError()
          << "requires " + Twine(extracts.size()) +
                 " qubits. However, the architecture only supports " +
                 Twine(device->nqubits()) + "qubits.";
      signalPassFailure();
      return;
    }

    auto layout = generateLayout(extracts);
    if (failed(layout)) {
      func->emitError() << "failed to refine random initial layouts.";
      signalPassFailure();
      return;
    }

    Statistics stats;
    RoutingBundle bundle{
        .wires = place(body, *layout, rewriter),
        .layout = std::move(*layout),
    };

    const auto res = route<WireDirection::Forward, RoutingMode::Hot>(
        bundle, stats, &rewriter);
    if (res.failed()) {
      func.emitError() << "failed to map the function";
      signalPassFailure();
      return;
    }

    // Collect statistics.
    numSwaps += stats.nswaps;

    // Fix SSA Dominance issues.
    for_each(body.getBlocks(), [](Block& b) { sortTopologically(&b); });
  }

private:
  /// Extend the init arguments of an `scf::ForOp` by adding a given range of
  /// additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static scf::ForOp extend(scf::ForOp forOp, ValueRange addons,
                           IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp);

    const auto res =
        forOp.replaceWithAdditionalIterOperands(rewriter, addons, true);
    assert(succeeded(res));
    auto newForOp = cast<scf::ForOp>(*res);

    for (const auto [before, after] : llvm::zip_equal(
             addons, newForOp.getResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newForOp);
    }
    return newForOp;
  }

  /// Extend the qubit arguments of an `IfOp` by adding a given range of
  /// additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static IfOp extend(IfOp ifOp, ValueRange addons, IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);

    auto newIfOp = ifOp.replaceWithAdditionalQubits(rewriter, addons);

    for (const auto [before, after] : llvm::zip_equal(
             addons, newIfOp->getResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newIfOp);
    }

    return newIfOp;
  }

  /// Extend the target arguments of an `IndexSwitchOp` by adding a given range
  /// of additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static IndexSwitchOp extend(IndexSwitchOp switchOp, ValueRange addons,
                              IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(switchOp);

    auto newSwitchOp = switchOp.replaceWithAdditionalTargets(rewriter, addons);
    for (const auto [before, after] : llvm::zip_equal(
             addons, newSwitchOp.getLinearResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newSwitchOp);
    }
    return newSwitchOp;
  }

  /// Extend the arguments of an `scf::WhileOp` by adding a given range of
  /// additional SSA values. Replaces the existing operation and returns the
  /// newly created one.
  static scf::WhileOp extend(scf::WhileOp whileOp, ValueRange addons,
                             IRRewriter& rewriter) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(whileOp);

    Block* oldBefBlock = whileOp.getBeforeBody();
    Block* oldAftBlock = whileOp.getAfterBody();

    const auto oldBefNumArgs = oldBefBlock->getNumArguments();
    const auto oldAftNumArgs = oldAftBlock->getNumArguments();

    // Create a new while op at the same location as the old one with the
    // additional arguments.

    SmallVector<Value> newInits(whileOp.getInits());
    newInits.append(addons.begin(), addons.end());

    SmallVector<Type> newTypes(whileOp.getResultTypes());
    newTypes.append(addons.getTypes().begin(), addons.getTypes().end());

    auto newWhileOp =
        rewriter.create<scf::WhileOp>(whileOp.getLoc(), newTypes, newInits);

    const SmallVector<Location> beforeLocs(newInits.size(), whileOp.getLoc());
    const SmallVector<Location> afterLocs(newTypes.size(), whileOp.getLoc());
    Block* newBefBlock =
        rewriter.createBlock(&newWhileOp.getBefore(), {},
                             ValueRange(newInits).getTypes(), beforeLocs);
    Block* newAftBlock =
        rewriter.createBlock(&newWhileOp.getAfter(), {}, newTypes, afterLocs);

    rewriter.mergeBlocks(oldBefBlock, newBefBlock,
                         newBefBlock->getArguments().take_front(oldBefNumArgs));
    rewriter.mergeBlocks(oldAftBlock, newAftBlock,
                         newAftBlock->getArguments().take_front(oldAftNumArgs));

    auto conditionOp = cast<scf::ConditionOp>(newBefBlock->getTerminator());
    rewriter.setInsertionPoint(conditionOp);

    // Replace the old condition operation with one that includes the new
    // "before" block arguments.

    SmallVector<Value> newConditionArgs(conditionOp.getArgs());
    llvm::append_range(newConditionArgs,
                       newBefBlock->getArguments().drop_front(oldBefNumArgs));

    rewriter.create<scf::ConditionOp>(
        conditionOp.getLoc(), conditionOp.getCondition(), newConditionArgs);
    rewriter.eraseOp(conditionOp);

    // Replace the old yield operation with one that includes the new "after"
    // block arguments.

    auto yieldOp = cast<scf::YieldOp>(newAftBlock->getTerminator());
    rewriter.setInsertionPoint(yieldOp);

    SmallVector<Value> newYieldArgs(yieldOp.getResults());
    llvm::append_range(newYieldArgs,
                       newAftBlock->getArguments().drop_front(oldAftNumArgs));

    rewriter.create<scf::YieldOp>(yieldOp.getLoc(), newYieldArgs);
    rewriter.eraseOp(yieldOp);

    // Finally, replace the old while operation with the new one.

    rewriter.replaceOp(
        whileOp, newWhileOp.getResults().take_front(whileOp.getNumResults()));

    for (const auto [before, after] : llvm::zip_equal(
             addons, newWhileOp->getResults().take_back(addons.size()))) {
      rewriter.replaceAllUsesExcept(before, after, newWhileOp);
    }

    return newWhileOp;
  }

  /// Return the wires of a dynamic computation.
  /// The mapping pass currently assumes that
  /// - there are no `qco.alloc` operation
  /// - there is an "extraction" and "insertion" phase, where the i-th extract
  ///   defines the i-th program qubit
  /// Thus, supported programs have the following structure:
  ///
  ///   T ⨉ [qtensor::AllocOp]
  /// → N ⨉ [qtensor::ExtractOp]
  /// → (Computation)
  /// → N ⨉ [qtensor::InsertOp]
  /// → T ⨉ [qtensor::DeallocOp]
  ///
  /// If any of the above assumptions are violated, the function returns
  /// failure.
  static FailureOr<SmallVector<Value>> getComputation(func::FuncOp func) {
    if (!func.getOps<AllocOp>().empty()) {
      return func.emitError() << "must not contain qco.alloc operations";
    }

    SmallVector<Value> extracts;

    for (auto alloc : func.getOps<qtensor::AllocOp>()) {
      bool isInitPhase = true;
      TensorIterator it(alloc.getResult());
      for (; it != std::default_sentinel; ++it) {
        if (auto extract = dyn_cast<ExtractOp>(it.operation())) {
          if (!isInitPhase) {
            return func.emitError()
                   << "must extract and insert all qubits at once.";
          }
          extracts.emplace_back(extract.getResult());
          continue;
        }

        if (isa<InsertOp>(it.operation())) {
          isInitPhase = false;
          continue;
        }
      }
    }

    return extracts;
  }

  /// Perform placement by
  /// - initializing as many hardware qubits as the architecture supports
  /// - replacing dynamic with static qubits
  /// - extending the inputs of `scf::ForOp` to all hardware qubits.
  ///
  /// Analogously to the getComputation function, the i-th extract
  /// operation defines the i-th program qubit.
  static Wires place(Region& body, const Layout& layout, IRRewriter& rewriter) {
    const auto nHw = layout.nHardwareQubits();
    SmallVector<StaticOp> staticOps;
    staticOps.reserve(nHw);

    // Create and save static qubit operations.
    rewriter.setInsertionPointToStart(&body.front());
    for (size_t i = 0; i < nHw; ++i) {
      const auto op = StaticOp::create(rewriter, body.getLoc(), i);
      staticOps.emplace_back(op);
      rewriter.setInsertionPointAfter(op);
    }

    // Preallocate one wire per hardware qubit.
    // Wire indices coincide with hardware qubit indices from here on.
    Wires wires;
    wires.reserve(nHw);
    for (size_t i = 0; i < nHw; ++i) {
      wires.emplace_back(staticOps[i].getQubit());
    }

    // Replace extract ops. Wires are already allocated, so the extract loop
    // only rewrites the IR.
    size_t nExtracts = 0;
    for (auto alloc : make_early_inc_range(body.getOps<qtensor::AllocOp>())) {
      TensorIterator it(alloc.getResult());
      while (it != std::default_sentinel) {
        // Get the operation and early increment to avoid issues after erasure.
        Operation* curr = it.operation();
        ++it;

        TypeSwitch<Operation*>(curr)
            .Case<ExtractOp>([&](auto op) {
              const auto prog = nExtracts++;
              const auto hw = layout.getHardwareIndex(prog);
              const auto qubit = staticOps[hw].getQubit();

              rewriter.replaceAllUsesWith(op.getResult(), qubit);
              rewriter.replaceAllUsesWith(op.getOutTensor(), op.getTensor());
              rewriter.eraseOp(op);
            })
            .Case<InsertOp>([&](auto op) {
              rewriter.setInsertionPointAfter(op);
              SinkOp::create(rewriter, op.getLoc(), op.getScalar());
              rewriter.replaceAllUsesWith(op.getResult(), op.getDest());
              rewriter.eraseOp(op);
            })
            .Case<DeallocOp>([&](auto op) { rewriter.eraseOp(op); });
      }

      rewriter.eraseOp(alloc);
    }

    // Sink the hardware qubits that no program qubit was placed on.
    rewriter.setInsertionPoint(body.back().getTerminator());
    for (size_t hw = 0; hw < nHw; ++hw) {
      if (layout.hasProgramAt(hw)) {
        continue;
      }
      SinkOp::create(rewriter, body.getLoc(), staticOps[hw].getQubit());
    }

    // Finally, update the SCF operations such that they take all static qubits
    // as input. To handle recursively nested SCF operations, use a stack of
    // (region, mapping) pairs.

    SmallVector<std::pair<Region&, DenseSet<Value>>> stack;
    stack.emplace_back(body, DenseSet<Value>{});

    while (!stack.empty()) {
      for (auto [region, qubits] = stack.pop_back_val();
           Operation& op : make_early_inc_range(region.getOps())) {
        TypeSwitch<Operation*>(&op)
            .Case<StaticOp>(
                [&](StaticOp staticOp) { qubits.insert(staticOp.getQubit()); })
            .Case<UnitaryOpInterface>([&](UnitaryOpInterface& uOp) {
              for (const auto [pred, succ] : llvm::zip_equal(
                       uOp.getInputQubits(), uOp.getOutputQubits())) {
                qubits.insert(succ);
                qubits.erase(pred);
              }
            })
            .Case<scf::ForOp>([&](scf::ForOp forOp) {
              assert(qubits.size() == layout.nHardwareQubits());

              llvm::for_each(getQubitValues(forOp.getInits()),
                             [&](Value v) { qubits.erase(v); });

              auto newForOp = extend(forOp, to_vector(qubits), rewriter);
              for (const auto [init, result] : llvm::zip_equal(
                       newForOp.getInits(), *newForOp.getLoopResults())) {
                if (isa<QubitType>(init.getType())) {
                  qubits.insert(result);
                  qubits.erase(init);
                }
              }

              const auto regionQubits =
                  getQubitValues(newForOp.getRegionIterArgs());
              stack.emplace_back(
                  newForOp.getRegion(),
                  DenseSet<Value>(regionQubits.begin(), regionQubits.end()));
            })
            .Case<scf::WhileOp>([&](scf::WhileOp whileOp) {
              assert(qubits.size() == layout.nHardwareQubits());

              llvm::for_each(getQubitValues(whileOp.getInits()),
                             [&](Value v) { qubits.erase(v); });

              auto newWhileOp = extend(whileOp, to_vector(qubits), rewriter);
              for (const auto [init, result] : llvm::zip_equal(
                       newWhileOp.getInits(), newWhileOp.getResults())) {
                if (isa<QubitType>(init.getType())) {
                  qubits.insert(result);
                  qubits.erase(init);
                }
              }

              const auto beforeArgs =
                  getQubitValues(newWhileOp.getBeforeArguments());
              const auto afterArgs =
                  getQubitValues(newWhileOp.getAfterArguments());
              stack.emplace_back(
                  newWhileOp.getBefore(),
                  DenseSet<Value>(beforeArgs.begin(), beforeArgs.end()));
              stack.emplace_back(
                  newWhileOp.getAfter(),
                  DenseSet<Value>(afterArgs.begin(), afterArgs.end()));
            })
            .Case<IfOp>([&](IfOp ifOp) {
              assert(qubits.size() == layout.nHardwareQubits());

              llvm::for_each(ifOp.getQubits(),
                             [&](Value v) { qubits.erase(v); });

              auto newIfOp = extend(ifOp, to_vector(qubits), rewriter);

              for (const auto [qubit, result] : llvm::zip_equal(
                       newIfOp.getQubits(), newIfOp.getLinearResults())) {
                qubits.insert(result);
                qubits.erase(qubit);
              }

              const auto thenArgs = newIfOp.getThenRegion().getArguments();
              const auto elseArgs = newIfOp.getElseRegion().getArguments();
              stack.emplace_back(
                  newIfOp.getThenRegion(),
                  DenseSet<Value>(thenArgs.begin(), thenArgs.end()));
              stack.emplace_back(
                  newIfOp.getElseRegion(),
                  DenseSet<Value>(elseArgs.begin(), elseArgs.end()));
            })
            .Case<IndexSwitchOp>([&](IndexSwitchOp switchOp) {
              assert(qubits.size() == layout.nHardwareQubits());

              llvm::for_each(switchOp.getTargets(),
                             [&](Value value) { qubits.erase(value); });

              auto newSwitchOp = extend(switchOp, to_vector(qubits), rewriter);
              for (const auto [target, result] :
                   llvm::zip_equal(newSwitchOp.getTargets(),
                                   newSwitchOp.getLinearResults())) {
                qubits.insert(result);
                qubits.erase(target);
              }

              for (Region* region : newSwitchOp.getRegions()) {
                const auto args = region->getArguments();
                stack.emplace_back(*region,
                                   DenseSet<Value>(args.begin(), args.end()));
              }
            })
            .Case<ResetOp, MeasureOp>([&](auto resetOp) {
              qubits.insert(resetOp.getQubitOut());
              qubits.erase(resetOp.getQubitIn());
            })
            .Case<AllocOp, qtensor::AllocOp>([&](auto) {
              llvm::reportFatalInternalError("unexpected dynamic qubit alloc");
            });
      }
    }

    return wires;
  }

  /// Execute `ntrials` many (parallel) initial layout refinement trials and
  /// return the heuristically best one.
  ///
  /// The function uses the SABRE Approach to improve the initial layout:
  /// Traverse the layers of the program from left-to-right-to-left and
  /// cold-route along the way. Repeat this procedure "niterations" times and
  /// finally find the trial with the fewest SWAPs on the final backwards pass
  /// and return the respective layout.
  FailureOr<Layout> generateLayout(ArrayRef<Value> extracts) {
    std::mt19937_64 rng{seed};
    const auto nProg = extracts.size();
    const auto nHw = device->nqubits();

    struct Trial {
      RoutingBundle bundle;
      Statistics stats{};
      bool success{false};
    };

    // For each trial, allocate a hardware-indexed wire array and install the
    // extract iterator on the hardware qubit picked by the random layout.
    // Unplaced hardware slots stay as sentinel wires (skipped by iteration).
    const auto buildBundle = [&](Layout layout) {
      Wires wires(nHw);
      for (size_t prog = 0; prog < nProg; ++prog) {
        size_t hw = layout.getHardwareIndex(prog);
        Value qubit = extracts[prog];
        wires[hw] = WireIterator(qubit);
      }
      return RoutingBundle{
          .wires = std::move(wires),
          .layout = std::move(layout),
      };
    };

    SmallVector<Trial, 0> trials;
    trials.reserve(ntrials);
    for (size_t i = 0; i < ntrials; ++i) {
      trials.emplace_back(buildBundle(Layout::random(nProg, nHw, rng())));
    }

    parallelForEach(&getContext(), trials, [&, this](Trial& t) {
      for (size_t i = 0; i < niterations; ++i) {
        if (route<WireDirection::Forward>(t.bundle, t.stats).failed()) {
          return;
        }
        t.stats.nswaps = 0;
        if (route<WireDirection::Backward>(t.bundle, t.stats).failed()) {
          return;
        }
      }

      t.success = true;
    });

    Trial* best = nullptr;
    for (Trial& t : trials) {
      if (t.success &&
          (best == nullptr || best->stats.nswaps > t.stats.nswaps)) {
        best = &t;
      }
    }

    if (best == nullptr) {
      return failure();
    }

    return best->bundle.layout;
  }

  /// Perform A* search to find a sequence of SWAPs that makes all two-qubit ops
  /// inside the first layer executable.
  ///
  /// The iteration budget is b^{3} node expansions, i.e. roughly a depth-3
  /// search in a tree with branching factor b, where b is the product of the
  /// architecture's maximum qubit degree and the maximum number of two-qubit
  /// gates in any layer: `b = maxDegree × ⌈N/2⌉`. A hard cap prevents
  /// impractical runtimes on larger architectures.
  ///
  /// Returns `failure`, if the A* search fails.
  FailureOr<SmallVector<IndexPairType>> search(const Window& window,
                                               const Layout& layout) const {
    constexpr size_t cap = 25'000'000UL;

    const size_t b = device->maxDegree() * ((device->nqubits() + 1) / 2);
    const size_t budget = std::min(b * b * b, cap);

    const Parameters params{.alpha = alpha, .lambda = lambda};

    llvm::SpecificBumpPtrAllocator<Node> arena;
    llvm::PriorityQueue<Node*, std::vector<Node*>, Node::ComparePointer>
        frontier;

    // Early exit, if the root node is a goal node already.
    Node* root = std::construct_at(arena.Allocate(), layout);
    if (root->isGoal(window.front(), *device)) {
      return SmallVector<IndexPairType>{};
    }

    frontier.emplace(root);

    DenseMap<ArrayRef<size_t>, size_t> bestDepth;
    // Owns the materialized layout snapshots that `bestDepth` keys point into.
    // std::deque never invalidates pointers to existing elements on push_back,
    // so ArrayRefs already in `bestDepth` stay stable as we grow it.
    std::deque<SmallVector<size_t>> keyStorage;
    SmallVector<IndexPairType, 6> expansionSet;

    const auto materializeKey = [](const Layout& layout) {
      SmallVector<size_t> key(layout.nProgramQubits());
      for (size_t prog = 0; prog < layout.nProgramQubits(); ++prog) {
        key[prog] = layout.getHardwareIndex(prog);
      }
      return key;
    };

    size_t i = 0;
    while (!frontier.empty() && i < budget) {
      Node* curr = frontier.top();
      frontier.pop();

      // Multiple sequences of SWAPs can lead to the same layout and the same
      // layout creates the same child-nodes. Thus, if we've seen a layout
      // already at a lower depth don't reexpand the current node (and hence
      // recreate the same child nodes).

      keyStorage.push_back(materializeKey(curr->layout));
      const auto [it, inserted] = bestDepth.try_emplace(
          ArrayRef<size_t>(keyStorage.back()), curr->depth);
      if (!inserted) {
        keyStorage.pop_back();
        if (const auto otherDepth = it->getSecond();
            curr->depth >= otherDepth) {
          ++i;
          continue;
        }

        it->second = curr->depth;
      }

      // If the currently visited node is a goal node, reconstruct the
      // sequence of SWAPs from this node to the root.

      if (curr->isGoal(window.front(), *device)) {
        SmallVector<IndexPairType> seq(curr->depth);
        size_t j = seq.size() - 1;
        for (const Node* n = curr; n->parent != nullptr; n = n->parent) {
          seq[j] = n->swap;
          --j;
        }

        return seq;
      }

      // Given a layout, create child-nodes for each possible SWAP
      // between two neighboring hardware qubits.

      expansionSet.clear();
      for (const auto& [q0, q1] = window.front(); const auto prog : {q0, q1}) {
        for (const auto hw0 = curr->layout.getHardwareIndex(prog);
             const auto hw1 : device->neighboursOf(hw0)) {
          // Ensure consistent hashing/comparison.
          const IndexPairType swap = std::minmax(hw0, hw1);
          if (is_contained(expansionSet, swap)) {
            continue;
          }
          expansionSet.push_back(swap);

          frontier.emplace(std::construct_at(arena.Allocate(), curr, swap,
                                             window, *device, params));
        }
      }

      ++i;
    }

    return failure();
  }

  /// Return the SWAP sequence to move from one layout to another.
  /// Implements the 4-Approximation algorithm described in arXiv:1602.05150v3.
  [[nodiscard]] SmallVector<IndexPairType> restore(const Layout& from,
                                                   const Layout& to) const {
    Layout curr(from);
    FGraph f(device);
    SmallVector<IndexPairType> swaps;

    while (true) {
      f.reset();
      f.construct(curr, to);

      if (const auto happy = f.findHappySWAPChain()) {
        for (const auto& swap : *happy) {
          swaps.emplace_back(swap);
          curr.swap(swap.first, swap.second);
        }
        continue;
      }

      // If there are no happy or unhappy swaps anymore,
      // the final placement of every token is reached.

      const auto unhappy = f.findUnhappySWAP();
      if (!unhappy) {
        break;
      }

      swaps.emplace_back(*unhappy);
      curr.swap(unhappy->first, unhappy->second);
    }

    return swaps;
  }

  /// Return a pair of SWAP sequences to transform two layouts into each other.
  /// Inspired by the 4-Approximation algorithm described in arXiv:1602.05150v3,
  /// with the key difference that the goal permutation is not static.
  [[nodiscard]] std::tuple<Layout, SmallVector<IndexPairType>,
                           SmallVector<IndexPairType>>
  converge(const Layout& lhs, const Layout& rhs) const {
    std::array layouts{Layout(lhs), Layout(rhs)};
    std::array graphs{FGraph(device), FGraph(device)};
    std::array<SmallVector<IndexPairType>, 2> swaps{};

    std::mt19937 gen(seed);
    std::uniform_int_distribution coin(0, 1);

    while (true) {
      size_t i = 0;
      for (; i < 2; ++i) {
        FGraph& f = graphs[i];

        f.reset();
        f.construct(layouts[i], layouts[(i + 1) % 2]);

        if (const auto happy = f.findHappySWAPChain()) {
          for (const auto& swap : *happy) {
            swaps[i].emplace_back(swap);
            layouts[i].swap(swap.first, swap.second);
          }
          break;
        }
      }

      // If we exit early from the loop, we've found a happy SWAP chain.
      if (i != 2) {
        continue;
      }

      // Otherwise, we randomly apply an unhappy SWAP to one of the layouts.
      // If there is no happy or unhappy swaps anymore, the final placement of
      // every token is reached.

      i = coin(gen);

      const auto unhappy = graphs[i].findUnhappySWAP();
      if (!unhappy) {
        break;
      }

      swaps[i].emplace_back(*unhappy);
      layouts[i].swap(unhappy->first, unhappy->second);
    }

    assert(layouts[0] == layouts[1]);

    return {layouts[0], std::move(swaps[0]), std::move(swaps[1])};
  }

  /// Skip to the end of the two-qubit block for both wire iterators, where
  /// initially both must point at the same two-qubit operation.
  template <WireDirection Direction>
  static void skipQubitPairBlock(WireIterator& it0, WireIterator& it1) {
    using Traits = WireTraversalTraits<Direction>;

    // Traverses the pair of wire iterators in tandem until a two-qubit
    // operation is found. If the two-qubit operation is equivalent, continue.
    // Otherwise, stop.

    std::array block{it0, it1};
    while (true) {
      for (auto& it : block) {
        while (Traits::isActive(it)) {
          std::ranges::advance(it, Traits::stride());

          if (it.operation() == nullptr) { // isa<Blockargument>
            return;
          }

          if (auto u = dyn_cast<UnitaryOpInterface>(it.operation());
              u && u.getNumQubits() > 1) {
            // Handle two-qubit barrier edge case explicitly.
            if (isa<BarrierOp>(u) && u.getNumQubits() != 2) {
              return;
            }
            // Otherwise stop for subsequent two-qubit unitary comparison.
            break;
          }
        }

        if (it == std::default_sentinel) {
          return;
        }
      }

      if (block[0].operation() != block[1].operation()) {
        return;
      }

      it0 = block[0];
      it1 = block[1];
    }
  }

  /// Return a window of layers with a maximum size of `1 + nlookahead`.
  template <WireDirection Direction>
  Window getWindow(Wires wires, const Layout& layout) {
    Window window;
    window.reserve(1 + nlookahead);

    walkProgramGraph<Direction>(
        wires, [&](const ReadyMap& ready, ReleasedOps& released) {
          if (ready.empty()) {
            return WalkResult::advance();
          }

          for (const auto& [op, indices] : ready) {
            if (isa<UnitaryOpInterface>(op)) {
              const auto hw0 = indices[0];
              const auto hw1 = indices[1];
              const auto prog0 = layout.getProgramIndex(hw0);
              const auto prog1 = layout.getProgramIndex(hw1);

              window.emplace_back(prog0, prog1);
              if (window.size() == 1 + nlookahead) {
                return WalkResult::interrupt();
              }

              skipQubitPairBlock<Direction>(wires[hw0], wires[hw1]);
              released.emplace_back(op);
              return WalkResult::advance();
            }

            released.emplace_back(op);
            return WalkResult::advance();
          }

          return WalkResult::advance();
        });

    return window;
  }

  /// Insert SWAP operations, exchanging two qubits, virtually
  /// (`RoutingMode::Cold`) or into the IR (`RoutingMode::Hot`).
  ///
  /// The wire index always equals the hardware qubit index, so wires are
  /// accessed by hardware index directly.
  ///
  /// In Hot mode the SWAP is written into the IR and the two wire iterators
  /// advance past it; walking each wire after the SWAP then follows the state
  /// that was moved into its hardware slot.
  ///
  /// In Cold mode there is no IR mutation, so swapping the two wire iterators
  /// in place mirrors the same effect: `wires[hw]` now walks the state that
  /// just moved onto `hw`.
  ///
  /// Both modes call `layout.swap` to update the program-to-hardware mapping.
  template <RoutingMode Mode>
  static void insertSWAPs(ArrayRef<IndexPairType> swaps, RoutingBundle& bundle,
                          Statistics& stats, IRRewriter* rewriter) {
    auto& [wires, layout] = bundle;
    for (const auto& [hw0, hw1] : swaps) {
      if constexpr (Mode == RoutingMode::Hot) {
        auto& w0 = wires[hw0];
        auto& w1 = wires[hw1];

        const auto in0 = w0.qubit();
        const auto in1 = w1.qubit();

        rewriter->setInsertionPointAfterValue(in0); // Valid bc. Hot => Forward.
        auto swapOp = SWAPOp::create(*rewriter, in0.getLoc(), in0, in1);

        const auto out0 = swapOp.getQubit0Out();
        const auto out1 = swapOp.getQubit1Out();

        rewriter->replaceAllUsesExcept(in0, out1, swapOp);
        rewriter->replaceAllUsesExcept(in1, out0, swapOp);

        std::advance(w0, 1); // Move to SWAP.
        std::advance(w1, 1);
      } else { // Mode == RoutingMode::Cold
        std::swap(wires[hw0], wires[hw1]);
      }

      layout.swap(hw0, hw1);
    }

    stats.nswaps += swaps.size();
  }

  /// Advance past all executable gates and return operations with nested
  /// regions and the respective wire indices. Stops when no more executable
  /// gates are found. After the function returns, the wires point at the
  /// results of non-executable gates or operations with nested regions.
  template <WireDirection Direction>
  RecursiveRoutingStack advance(Wires& wires) {
    RecursiveRoutingStack stack;
    DenseSet<Operation*> visited;

    // Advance wires past all executable gates and push operations with
    // nested regions and the respective wire indices of their inputs onto the
    // result stack.

    walkProgramGraph<Direction>(
        wires, [&](const ReadyMap& ready, ReleasedOps& released) {
          if (ready.empty()) {
            return WalkResult::advance();
          }

          for (const auto& [readyOp, indices] : ready) {
            TypeSwitch<Operation*>(readyOp)
                .template Case<BarrierOp>(
                    [&](BarrierOp op) { released.emplace_back(op); })
                .template Case<UnitaryOpInterface>([&](UnitaryOpInterface op) {
                  // Under the wire-index-equals-hardware-index invariant, the
                  // walk indices are already the hardware qubits.
                  if (device->areAdjacent(indices[0], indices[1])) {
                    released.emplace_back(op);
                  }
                })
                .template Case<scf::ForOp, scf::WhileOp, IfOp, IndexSwitchOp>(
                    [&](auto op) {
                      if (visited.insert(op).second) {
                        stack.emplace_back(op, indices);
                      }
                    });
          }

          if (released.empty()) {
            return WalkResult::interrupt();
          }

          return WalkResult::advance();
        });

    return stack;
  }

  /// Return `values` with only the qubit entries realigned according to the
  /// given permutation of hardware indices.
  static SmallVector<Value> realignQubitValues(ValueRange values,
                                               ArrayRef<size_t> perm,
                                               const RoutingBundle& bundle) {
    // Index the value map directly by wire position; under the wire-index-
    // equals-hardware-index invariant, position `i` is the hardware qubit at
    // position `i`.
    DenseMap<size_t, Value> m(bundle.wires.size());
    for (size_t i = 0; i < bundle.wires.size(); ++i) {
      m.try_emplace(i, bundle.wires[i].qubit());
    }

    SmallVector<Value> realigned(values);
    size_t qubitIndex = 0;
    for (Value& value : realigned) {
      if (isa<QubitType>(value.getType())) {
        value = m.at(perm[qubitIndex++]);
      }
    }
    assert(qubitIndex == perm.size());
    return realigned;
  }

  /// Processes the recursive stack item by routing the nested operation and
  /// inserting epilogue SWAPs.
  template <WireDirection Direction, RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  LogicalResult dispatch(const RecursiveRoutingStackItem& item,
                         RoutingBundle& parent, Statistics& stats,
                         IRRewriter* rewriter = nullptr) {
    const auto& [op, indices] = item;
    const auto nHw = parent.wires.size();

    // `permutation[qubitResNum] = hw` records at which hardware qubit the
    // SCF's qubit-only operand at position `qubitResNum` was passed.
    // `place()` extends SCF ops via a `DenseSet<Value>` whose iteration order
    // does not follow hardware indices, so this mapping is needed to:
    //   (a) realign each child region's terminator (WhileOp/IfOp/IndexSwitchOp)
    //       back into hardware order, and
    //   (b) walk WhileOp's after-region in hardware order when populating
    //      `children[1].wires`.
    SmallVector<size_t> permutation(indices.size());
    SmallVector<RoutingBundle, 2> children =
        TypeSwitch<Operation*, SmallVector<RoutingBundle, 2>>(op)
            .template Case<scf::ForOp, scf::WhileOp>([&](auto) {
              return SmallVector<RoutingBundle, 2>{
                  RoutingBundle{.wires = Wires(nHw), .layout = parent.layout},
              };
            })
            .template Case<IfOp>([&](IfOp) {
              return SmallVector<RoutingBundle, 2>{
                  RoutingBundle{.wires = Wires(nHw), .layout = parent.layout},
                  RoutingBundle{.wires = Wires(nHw), .layout = parent.layout}};
            })
            .template Case<IndexSwitchOp>([&](IndexSwitchOp switchOp) {
              return SmallVector<RoutingBundle, 2>(
                  switchOp.getNumRegions(),
                  RoutingBundle{.wires = Wires(nHw), .layout = parent.layout});
            });

    SmallVector<std::optional<size_t>> resultToQubitIndex(op->getNumResults());
    size_t numQubitResults = 0;
    for (const auto [resultIndex, result] : llvm::enumerate(op->getResults())) {
      if (isa<QubitType>(result.getType())) {
        resultToQubitIndex[resultIndex] = numQubitResults++;
      }
    }
    assert(numQubitResults == indices.size());

    SmallVector<Value> whileBeforeQubits;
    SmallVector<Value> whileConditionQubits;
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      whileBeforeQubits = getQubitValues(whileOp.getBeforeArguments());
      whileConditionQubits = getQubitValues(
          cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator())
              .getArgs());
    }

    // For each hardware qubit that reaches the SCF op, install a wire on the
    // matching child block argument.
    // Under the wire-index-equals-hardware-index invariant the child's wire at
    // position `i` walks that hardware qubit's SSA chain inside the child
    // region.
    for (size_t i : indices) {
      const auto res = cast<OpResult>(parent.wires[i].qubit());
      const auto resNum = res.getResultNumber();
      const auto qubitResNum = *resultToQubitIndex[resNum];

      TypeSwitch<Operation*>(op)
          .template Case<scf::ForOp>([&](scf::ForOp forOp) {
            children[0].wires[i] = WireIterator([&] -> Value {
              const auto arg = forOp.getTiedLoopRegionIterArg(res);
              if constexpr (Direction == WireDirection::Forward) {
                return arg;
              } else {
                return forOp.getTiedLoopYieldedValue(arg)->get();
              }
            }());
          })
          .template Case<scf::WhileOp>([&](scf::WhileOp) {
            children[0].wires[i] = WireIterator([&] -> Value {
              const auto arg = whileBeforeQubits[qubitResNum];
              if constexpr (Direction == WireDirection::Forward) {
                return arg;
              } else {
                return whileConditionQubits[qubitResNum];
              }
            }());
          })
          .template Case<IfOp>([&](IfOp ifOp) {
            assert(ifOp.getNumRegions() == 2);

            OpOperand* const qubit = ifOp.getTiedQubit(res);
            for (size_t r = 0; r < 2; ++r) {
              Region& region = ifOp.getRegion(r);

              const auto arg = region.getRegionNumber() == 0
                                   ? ifOp.getTiedThenBlockArgument(qubit)
                                   : ifOp.getTiedElseBlockArgument(qubit);
              children[r].wires[i] = WireIterator([&] -> Value {
                if constexpr (Direction == WireDirection::Forward) {
                  return arg;
                } else {
                  return region.getRegionNumber() == 0
                             ? ifOp.getTiedThenYieldedValue(arg)->get()
                             : ifOp.getTiedElseYieldedValue(arg)->get();
                }
              }());
            }
          })
          .template Case<IndexSwitchOp>([&](IndexSwitchOp switchOp) {
            OpOperand* const target = switchOp.getTiedTarget(res);
            for (Region* region : switchOp.getRegions()) {
              const auto regionNumber = region->getRegionNumber();
              const auto arg =
                  regionNumber == 0
                      ? switchOp.getTiedDefaultBlockArgument(target)
                      : switchOp.getTiedCaseBlockArgument(target,
                                                          regionNumber - 1);
              children[regionNumber].wires[i] = WireIterator([&] -> Value {
                if constexpr (Direction == WireDirection::Forward) {
                  return arg;
                }
                return regionNumber == 0
                           ? switchOp.getTiedDefaultYieldedValue(arg)->get()
                           : switchOp
                                 .getTiedCaseYieldedValue(arg, regionNumber - 1)
                                 ->get();
              }());
            }
          });

      permutation[qubitResNum] = i;
    }

    // Route each child, then restore its layout back to parent's.
    // Restoring every child gives the SCF a single well-defined exit layout
    // equal to the entry layout, which preserves the position-to-program
    // mapping that parent-scope ops consuming the SCF's results rely on.
    // TODO: Parallelize multiple children, if possible.

    const auto finishChild = [&](RoutingBundle& child) -> LogicalResult {
      if (failed(route<Direction, Mode>(child, stats, rewriter))) {
        return failure();
      }
      if constexpr (Mode == RoutingMode::Hot) {
        for_each(child.wires, [](auto& it) { std::advance(it, -2); });
      }
      const auto swaps = restore(child.layout, parent.layout);
      insertSWAPs<Mode>(swaps, child, stats, rewriter);
      return success();
    };

    for (auto& child : children) {
      if (failed(finishChild(child))) {
        return failure();
      }
    }

    // The WhileOp's after region is set up now, after children[0] (the before
    // region) has been restored to parent.layout. This way `permutation`,
    // which is indexed against `parent.layout`, correctly identifies the
    // hardware qubit for each after-block argument.

    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      children.emplace_back(RoutingBundle{
          .wires = Wires(nHw),
          .layout = parent.layout,
      });
      assert(children.size() == 2);

      const auto values = [&] -> ValueRange {
        if constexpr (Direction == WireDirection::Forward) {
          return whileOp.getAfterArguments();
        }
        return cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator())
            .getResults();
      }();
      const auto rng = getQubitValues(values);

      for (const auto& [k, arg] : llvm::enumerate(rng)) {
        children[1].wires[permutation[k]] = WireIterator(arg);
      }

      if (failed(finishChild(children[1]))) {
        return failure();
      }
    }

    // All children have been restored to `parent.layout`, so the SCF's exit
    // layout equals parent's entry layout.
    const Layout exit = parent.layout;

    if constexpr (Mode == RoutingMode::Hot) {

      // Realign terminator values to ensure that i-th input qubit and the i-th
      // output qubit represent the equivalent hardware qubit.

      TypeSwitch<Operation*>(op)
          .template Case<scf::ForOp>([&](scf::ForOp forOp) {
            auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            rewriter->setInsertionPoint(yieldOp);
            rewriter->replaceOpWithNewOp<scf::YieldOp>(
                yieldOp, realignQubitValues(yieldOp.getResults(), permutation,
                                            children[0]));
          })
          .template Case<scf::WhileOp>([&](scf::WhileOp whileOp) {
            auto condOp = cast<scf::ConditionOp>(
                whileOp.getBeforeBody()->getTerminator());
            rewriter->setInsertionPoint(condOp);
            rewriter->replaceOpWithNewOp<scf::ConditionOp>(
                condOp, condOp.getCondition(),
                realignQubitValues(condOp.getArgs(), permutation, children[0]));

            auto yieldOp =
                cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator());
            rewriter->setInsertionPoint(yieldOp);
            rewriter->replaceOpWithNewOp<scf::YieldOp>(
                yieldOp, realignQubitValues(yieldOp.getResults(), permutation,
                                            children[1]));
          })
          .template Case<IfOp, IndexSwitchOp>([&](auto branchOp) {
            for (const auto [region, child] :
                 llvm::zip_equal(branchOp->getRegions(), children)) {
              auto yieldOp = cast<YieldOp>(region.front().getTerminator());
              rewriter->setInsertionPoint(yieldOp);
              rewriter->replaceOpWithNewOp<YieldOp>(
                  yieldOp,
                  realignQubitValues(yieldOp.getTargets(), permutation, child));
            }
          });

      // Sort topologically to fix any occurring SSA dominance errors.

      for (Region& region : op->getRegions()) {
        assert(region.hasOneBlock());
        sortTopologically(&region.front());
      }
    }

    // Every SCF op restores its children to `parent.layout` at exit, so
    // `parent.layout` is unchanged by the dispatch and there is nothing to
    // sync. Move past the SCF op by incrementing every wire currently
    // sitting on it. Under Cold routing the wire container may have been
    // reshuffled by insertSWAPs, so match by op-identity rather than by the
    // original `indices` positions; this works in both modes.
    assert(exit == parent.layout);
    for (auto& wire : parent.wires) {
      if (wire.operation() == op) {
        std::advance(wire, WireTraversalTraits<Direction>::stride());
      }
    }

    return success();
  }

  /// Iterates over a dynamically computed window of layers and uses A* search
  /// to find a SWAP sequence that makes each layer executable. Depending on
  /// the template parameter, this function only updates the layout or also
  /// inserts the SWAPs into the IR. The function returns `failure` if A* is
  /// unable to find a solution.
  template <WireDirection Direction, RoutingMode Mode = RoutingMode::Cold>
    requires(Mode != RoutingMode::Hot || Direction == WireDirection::Forward)
  LogicalResult route(RoutingBundle& bundle, Statistics& stats,
                      IRRewriter* rewriter = nullptr) {
    auto& [wires, layout] = bundle;

    while (true) {

      while (true) {
        const auto stack = advance<Direction>(wires);
        if (stack.empty()) {
          break;
        }
        for (const auto& item : stack) {
          if (dispatch<Direction, Mode>(item, bundle, stats, rewriter)
                  .failed()) {
            return failure();
          }
        }
      }

      const auto window = getWindow<Direction>(wires, layout);
      if (window.empty()) {
        break;
      }

      const auto swaps = search(window, layout);
      if (failed(swaps)) {
        return failure();
      }

      if constexpr (Mode == RoutingMode::Hot) {

        // At this point the wire iterators either point to
        // std::default_sentinel or a multi-qubit gate (incl. barriers) of
        // the current or subsequent layers. The former must be decremented
        // twice (sentinel → sink → unitary/static). For the latter, we
        // must ensure the insertion point is before the multi-qubit gates.

        for (auto& it : wires) {
          std::advance(it, it == std::default_sentinel ? -2 : -1);
        }
      }

      insertSWAPs<Mode>(*swaps, bundle, stats, rewriter);

      if constexpr (Mode == RoutingMode::Hot) {

        // After SWAP insertion, a wire is either untouched by the SWAP
        // insertion or pointing at a SWAP operation. If the former is the
        // case, incrementing the wire iterator will undo the previous
        // decrement, leaving it at the same position as before the SWAP
        // insertion. Otherwise, an increment will move the iterator to the
        // multi-qubit op of the current or subsequent layer or to a sink (and
        // thus std::default_sentinel).

        for_each(wires, [](auto& it) { std::advance(it, 1); });
      }
    }

    return success();
  }

  std::shared_ptr<AugmentedDevice> device;
};

} // namespace

std::unique_ptr<Pass>
createMappingPass(const DenseSet<std::pair<size_t, size_t>>& couplingSet,
                  MappingPassOptions options) {

  // Verify the assumption that the coupling set is symmetric:
  // For every edge (u, v) in the set, (v, u) must also be present.

  for (const auto& [u, v] : couplingSet) {
    if (u == v) {
      llvm::reportFatalUsageError("Found an invalid (u, u) edge.");
    }

    if (!couplingSet.contains({v, u})) {
      llvm::reportFatalUsageError("Expected symmetric coupling set: edge (" +
                                  Twine(u) + ", " + Twine(v) +
                                  ") exists but (" + Twine(v) + ", " +
                                  Twine(u) + ") does not.");
    }
  }

  return std::make_unique<MappingPass>(couplingSet, options);
}

} // namespace mlir::qco
