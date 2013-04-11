//===------- StaticProfilePass.h - Interface to Static Profile Pass -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describe the interface to the static profile pass. It estimates
// function call, edges and basic block execution count (frequencies).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_STATIC_PROFILE_PASS_H
#define LLVM_ANALYSIS_STATIC_PROFILE_PASS_H

#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/ProfileInfo.h"
#include <vector>

namespace llvm {
  class CallGraph;
  class CallGraphNode;
  class Function;
  class BlockEdgeFrequencyPass;

  class StaticProfilePass : public ModulePass,
                            public ProfileInfo {
  private:
    CallGraph *CG;
    static double epsilon;

    typedef std::pair<const Function *, const Function *> FunctionEdge;
    DenseMap<FunctionEdge, double> LocalEdgeFrequency;
    DenseMap<FunctionEdge, double> GlobalEdgeFrequency;
    DenseMap<FunctionEdge, double> BackEdgeFrequency;
    DenseMap<const Function *, DenseSet<const Function *> > Predecessors;

    DenseSet<FunctionEdge> FunctionBackEdges;
    DenseSet<CallGraphNode *> FunctionLoopHeads;
    DenseSet<const Function *> NotVisited;

    std::vector<CallGraphNode *> DepthFirstOrder;

    /// Preprocess - From a call graph:
    ///   (1) obtain functions in depth-first order;
    ///   (2) find back edges;
    ///   (3) find loop heads;
    ///   (4) local block and edge profile information (per function);
    ///   (5) local function edge frequency;
    ///   (6) map of function predecessors.
    ///  Procedure based on FindFunctionBackedges inside BasicBlockUtils.
    void Preprocess();

    /// UpdateCallInfo - Calculates local function edges (function invocations)
    /// and a map of function predecessors.
    void UpdateCallInfo(Function &F, BlockEdgeFrequencyPass &BEFP);

    /// MarkReachable - Mark all blocks reachable from root function as not
    /// visited.
    void MarkReachable(CallGraphNode *root);

    /// PropagateCallFrequency - Calculate function call and invocation
    /// frequencies.
    void PropagateCallFrequency(CallGraphNode *node, bool end);

    /// CalculateGlobalInfo - With calculated function frequency, recalculate
    /// block and edge frequencies taking it into consideration.
    void CalculateGlobalInfo(Module &M);

    /// getBackEdgeFrequency - Get updated back edges frequency. In case of not
    /// found, use the local edge frequency.
    double getBackEdgeFrequency(FunctionEdge &fedge) const;
	
	//////////////////////
	void printFrequency( Function *F, raw_ostream &OS );
	int GetFuncCall(Module *module, CallGraph *CG);
  public:
    static char ID; // Class identification, replacement for typeinfo

    StaticProfilePass();
    ~StaticProfilePass();

    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual const char *getPassName() const;
    virtual bool runOnModule(Module &M);

    /// getGlobalEdgeFrequency - Get updated global edge frequency. In case of
    /// not found, use the local edge frequency.
    double getGlobalEdgeFrequency(FunctionEdge &fedge) const;
  };
}  // End of llvm namespace

#endif