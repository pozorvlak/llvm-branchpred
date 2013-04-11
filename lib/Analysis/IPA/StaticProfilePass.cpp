//===---- StaticProfilePass.cpp - LLVM Pass to Static Profile Modules -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass staticly estimates execution counts for blocks, edges an function
// calls invocations in compilation time. This algorithm is slightly modified
// from the BlockEdgeFrequencyPass to calculate cyclic_frequencies for
// functions in a call graph.
//
// References:
// Youfeng Wu and James R. Larus. Static branch frequency and program profile
// analysis. In MICRO 27: Proceedings of the 27th annual international symposium
// on Microarchitecture. IEEE, 1994.//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "static-profile"

#include "llvm/BasicBlock.h"
#include "llvm/InstrTypes.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Analysis/BlockEdgeFrequencyPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/StaticProfilePass.h"
//#include "llvm/Analysis/StaticCfg.h"
using namespace llvm;

const std::map<const Function *, std::map<const BasicBlock *, double> > *g_hF2B2Acc;
std::map<std::string, std::set<std::string> > g_hFuncCall;
bool g_bRecursive = false;

void printCFG(Function *function, raw_ostream &OS)
{
	OS << static_cast<Value&>(*function);
}

static void printPCG(raw_ostream &os)
{
    os << "digraph G {\n";
    std::map<std::string, std::set<std::string> >::iterator f2s_p = g_hFuncCall.begin(), f2s_e = g_hFuncCall.end();
    for(; f2s_p != f2s_e; ++f2s_p)
    {
        if( f2s_p->second.empty() )
            continue;
        std::set<std::string>::iterator f_p = f2s_p->second.begin(), f_e = f2s_p->second.end();
        for(; f_p != f_e; ++ f_p)
            os << f2s_p->first << " -> " << *f_p << "\n";
        os << "\n";
    }
    os << "}\n";

}
// add by qali: output block frequency estimation
void dumpFreq()
{
	std::map<const Function *, std::map<const BasicBlock *, double> >::const_iterator f2b2acc_p = g_hF2B2Acc->begin(), E = g_hF2B2Acc->end();
	for( ; f2b2acc_p  != E; ++ f2b2acc_p )
	{
		errs() << "\n" << f2b2acc_p->first->getName() << ":\n";
		std::map<const BasicBlock *, double>::const_iterator b2acc_p = f2b2acc_p->second.begin(), EE = f2b2acc_p->second.end();
		for(; b2acc_p != EE; ++ b2acc_p)
			errs() << b2acc_p->first->getName() << "\n";
	}
}
void StaticProfilePass::printFrequency( Function *F, raw_ostream &OS )
{
	dumpFreq();
	OS << "\n###################################################################\n";
	BlockCounts &blocks = BlockInformation[F];
    for (BlockCounts::iterator BI = blocks.begin(), BE = blocks.end();
         BI != BE; ++BI)
		 {
			 OS << "\n";
			 OS << F->getName() << "_";
			 OS << BI->first->getName();
			 OS << ":\t";
			 OS << BI->second;
		 }
	/*for (Function::const_iterator I = F->begin(), E = F->end(); I != E; ++I)
	{
		if (I->hasName())
		{              // Print out the label if it exists...
			OS << "\n";
			OS << F->getName() << "_";
			OS << I->getName();
			OS << ":\t";

			std::map<const Function *, std::map<const BasicBlock *, double> >::const_iterator f2b2acc_p, E = g_hF2B2Acc->end();
			if( (f2b2acc_p = g_hF2B2Acc->find(F) ) != E )
			{
				std::map<const BasicBlock *, double>::const_iterator b2acc_p, EE = f2b2acc_p->second.end();
				if( (b2acc_p = f2b2acc_p->second.find(I)) != EE )
					OS << b2acc_p->second;
			}
		}
	}*/
}

int GetAccess(const std::map<const Function *, std::map<const BasicBlock *, double> > &left)
{
	g_hF2B2Acc = &left;
	return 0;
}

// by qali: obtain the function call graph
int StaticProfilePass::GetFuncCall(Module *module, CallGraph *CG)
{
	if( g_bRecursive )
		return 0;

	CallGraphNode *root = CG->getRoot();
	SmallVector<CallGraphNode *, 48> stack;
	std::set<Function *> funcSet;

	// Added the function root into the stack.
	stack.push_back(root);
	funcSet.insert((root->getFunction()) );

	//////////////////////////////////
	std::string szTemp;
	std::string szCfg = module->getModuleIdentifier() + ".cfg";
	std::string szFreq = module->getModuleIdentifier() + ".freq";
	raw_fd_ostream cfgFile(szCfg.c_str(), szTemp, raw_fd_ostream::F_Append);
	raw_fd_ostream freqFile(szFreq.c_str(), szTemp, raw_fd_ostream::F_Append);

	if( root->getFunction())
	{
		printCFG(root->getFunction(), cfgFile);
		printFrequency(root->getFunction(), freqFile);
	}
	///////////////////////////////////////////

	while (!stack.empty()) {
		CallGraphNode *CGN = stack.pop_back_val();
		if (Function *F = CGN->getFunction()) {
			// Add successors to the stack for future processing.
			for (CallGraphNode::iterator CI = CGN->begin(), CE = CGN->end();
			   CI != CE; ++CI)
			{
				Function *Callee = CI->second->getFunction();
				if( Callee && !Callee->isDeclaration() )
				{
					if( funcSet.find(Callee) == funcSet.end())
					{
						stack.push_back(CI->second);
						printCFG(Callee, cfgFile);
						printFrequency(Callee, freqFile);
					}
					g_hFuncCall[F->getName()].insert(Callee->getName());
					funcSet.insert(Callee);
				}

			}
		}
	}
	cfgFile.close();
	freqFile.close();

    std::string szPCG = module->getModuleIdentifier() + ".dot1";
    raw_fd_ostream pcgFile(szPCG.c_str(), szTemp, raw_fd_ostream::F_Append);
    printPCG(pcgFile);
    pcgFile.close();


	std::set<Function *>::iterator K = funcSet.begin(), KE = funcSet.end();
	for( ; K != KE; ++K)
	{
		std::set<Function *>::iterator I = funcSet.begin(), E1 = funcSet.end();
		for(; I != E1; ++I)
		{
			std::set<Function *>::iterator J = funcSet.begin(), E2 = funcSet.end();
			for(; J != E2; ++ J)
			{
				std::set<std::string> & IRels = g_hFuncCall[(*I)->getName()];
				std::set<std::string> & KRels = g_hFuncCall[(*K)->getName()];
				if( IRels.find((*K)->getName()) != IRels.end() && KRels.find((*J)->getName()) != KRels.end() )
					g_hFuncCall[(*I)->getName()].insert((*J)->getName());
			}
		}
	}

	errs() << "-----------function call--------------------\n";
	std::map<std::string, std::set<std::string> >::iterator I = g_hFuncCall.begin(), E = g_hFuncCall.end();
	for( ; I != E; ++I)
	{
		errs() << I->first << ":\t";
		std::set<std::string>::iterator FI = I->second.begin(), FE = I->second.end();
		for(; FI != FE; ++FI )
			errs() << *FI << ",";
		errs() << "\n";
	}
	return 0;
}

char StaticProfilePass::ID = 0;
double StaticProfilePass::epsilon = 0.000001;

static RegisterPass<StaticProfilePass>
X("static-profile", "Static profile pass", false, true);

static RegisterAnalysisGroup<ProfileInfo> Y(X);

ModulePass *llvm::createStaticProfilePass() {
  return new StaticProfilePass();
}

StaticProfilePass::StaticProfilePass() : ModulePass(ID) {
}

StaticProfilePass::~StaticProfilePass() {
}

void StaticProfilePass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraph>();
  AU.addRequired<BlockEdgeFrequencyPass>();
  // add
  //AU.addRequired<LoopInfo>();
  //
  AU.setPreservesAll();
}

const char *StaticProfilePass::getPassName() const {
  return "Static profile";
}

bool StaticProfilePass::runOnModule(Module &M) {
  CG = &getAnalysis<CallGraph>();

  // Calculate necessary information before processing.
  Preprocess();

  // Search for function loop heads in reverse depth-first order.
  std::vector<CallGraphNode *>::reverse_iterator RI, RE;
  for (RI = DepthFirstOrder.rbegin(), RE = DepthFirstOrder.rend();
       RI != RE; ++RI) {
    CallGraphNode *node = *RI;

    // If function is a loop head, propagate frequencies from it.
    if (FunctionLoopHeads.count(node)) {
      // Mark all reachable nodes as not visited.
      MarkReachable(node);

      // Propagate call frequency starting from this loop head.
      PropagateCallFrequency(node, false);
    }
  }

  // Release some unused memory.
  DepthFirstOrder.clear();
  FunctionLoopHeads.clear();

  // Obtain the main function.
  CallGraphNode *root = CG->getRoot();

  // Mark all functions reachable from the main function as not visited.
  MarkReachable(root);

  // Propagate frequency starting from the main function.
  PropagateCallFrequency(root, true);

  // With function frequency calculated, propagate it to block and edge
  // frequencies to achieve global block and edge frequency.
  CalculateGlobalInfo(M);

	// by qali: obtain block frequency estimation and function call graph
	GetAccess(BlockInformation);
	GetFuncCall(&M, CG);

	// for spm
	// 1. array splitting at loop sites
	// 2. live analysis
	// 3. collect the max-cliques
	// 4. spill arrays from the cliques larger than the size of SPM
	// 5. unspill nonsplit arrays into the interference graph
	//
  return false;
}

/// getGlobalEdgeFrequency - Get updated global edge frequency. In case of not
/// found, use the local edge frequency.
double StaticProfilePass::getGlobalEdgeFrequency(FunctionEdge &fedge) const {
  DenseMap<FunctionEdge, double>::const_iterator I =
      GlobalEdgeFrequency.find(fedge);
  if (I != GlobalEdgeFrequency.end())
    return I->second;

  I = LocalEdgeFrequency.find(fedge);
  return (I != LocalEdgeFrequency.end() ? I->second : 0.0);
}

/// Preprocess - From a call graph:
///   (1) obtain functions in depth-first order;
///   (2) find back edges;
///   (3) find loop heads;
///   (4) local block and edge profile information (per function);
///   (5) local function edge frequency;
///   (6) map of function predecessors.
///  Based on FindFunctionBackedges inside BasicBlockUtils.
void StaticProfilePass::Preprocess() {
  // Start with the main function.
  CallGraphNode *Current = CG->getRoot();

  // If main has no successors, i.e., calls no other functions.
  // Or it does not point to a valid function.
  if (Current->begin() == Current->end() || !Current->getFunction())
    return;

  // Auxiliary data structures.
  DenseSet<CallGraphNode *> Visited;
  DenseSet<CallGraphNode *> InStack;
  std::vector<CallGraphNode *> VisitStack;

  Visited.insert(Current);
  InStack.insert(Current);
  VisitStack.push_back(Current);

  do {
    CallGraphNode *Parent = VisitStack.back();
    CallGraphNode::const_iterator I = Parent->begin(),
                                  E = Parent->end();

    // Use when found a successor not knew before.
    bool FoundNew = false;

    // Search function successors.
    while (I != E) {
      Current = I->second;
      ++I;

      if (Function *CurrentFunct = Current->getFunction()) {
        // Try to insert the function into the visit list.
        // In case of success, a new function was found.
        if ((Visited.insert(Current)).second) {
          FoundNew = true;
          break;
        }

        // If successor is in VisitStack, it is a back edge.
        if (InStack.count(Current)) {
          // Save the function back edge.
		  llvm::errs() << "Recursive function calling for " << CurrentFunct->getName() << "\n";
		  g_bRecursive = true;
          FunctionBackEdges.insert(
            std::make_pair(Parent->getFunction(), CurrentFunct)
          );

          // Consider a loop head the function pointing by this back edge.
          FunctionLoopHeads.insert(Current);
        }
      }
    }

    // Found no new function, process it.
    if (!FoundNew) {
      // Obtain the function without new successors.
      CallGraphNode *Node = VisitStack.back();
      Function *F = Node->getFunction();
      VisitStack.pop_back();

      // Save this function ordering position (in depth-first order).
      DepthFirstOrder.push_back(Node);

      // Only process if it has a function body.
      if (!F->isDeclaration()) {
        // Calculate local block and edge frequencies.
        BlockEdgeFrequencyPass &BEFP =
            getAnalysis<BlockEdgeFrequencyPass>(*F);

        // Find all block frequencies.
        BlockEdgeFrequencyPass::block_iterator BFI, BFE;
        for (BFI = BEFP.block_freq_begin(), BFE = BEFP.block_freq_end();
             BFI != BFE; ++BFI)
          BlockInformation[F][BFI->first] = BFI->second;

        // Find all edge frequencies.
        BlockEdgeFrequencyPass::edge_iterator EFI, EFE;
        for (EFI = BEFP.edge_freq_begin(), EFE = BEFP.edge_freq_end();
             EFI != EFE; ++EFI)
          EdgeInformation[F][EFI->first] = EFI->second;

        // Update call information.
        UpdateCallInfo(*F, BEFP);
      }

      // Reach the bottom, go one level up.
      InStack.erase(Node);
    } else { // Found a new function.
      // Go down one level if there is a unvisited successor.
      InStack.insert(Current);
      VisitStack.push_back(Current);
    }
  } while (!VisitStack.empty());
}

/// UpdateCallInfo - Calculates local function edges (function invocations)
/// and a map of function predecessors.
void StaticProfilePass::UpdateCallInfo(Function &F,
                                       BlockEdgeFrequencyPass &BEFP) {
  DEBUG(errs() << "UpdateCallInfo(" << F.getName() << ", ...)\n");

  // Search for function invocations inside basic blocks.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    BasicBlock *BB = FI;
    double bfreq = BEFP.getBlockFrequency(BB);

    // Run over through all basic block searching for call instructions.
    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end();
         BI != BE; ++BI) {
      // Check if the instruction is a call instruction.
      if (CallInst *CI = dyn_cast<CallInst>(BI)) {
        if (Function *called = CI->getCalledFunction()) {
          FunctionEdge fedge = std::make_pair(&F, called);

          // The local edge frequency is the sum of block frequency from all
          // blocks that calls another function.
          LocalEdgeFrequency[fedge] += bfreq;

          // Define the predecessor of this function.
          Predecessors[called].insert(&F);

          // Print local function edge frequency.
          DEBUG(errs() << "  LocalEdgeFrequency[(" << F.getName() << ", "
                       << called->getName() << ")] = "
                       << format("%.3f", LocalEdgeFrequency[fedge]) << "\n");
        }
      }
    }
  }
}

/// MarkReachable - Mark all blocks reachable from root function as not visited.
void StaticProfilePass::MarkReachable(CallGraphNode *root) {
  // Clear the list first.
  NotVisited.clear();

  // Use a stack to search function successors.
  SmallVector<CallGraphNode *, 16> stack;

  // Added the function root into the stack.
  stack.push_back(root);

  while (!stack.empty()) {
    // Retrieve a function from the stack to process.
    CallGraphNode *CGN = stack.pop_back_val();
    if (Function *F = CGN->getFunction()) {
      // If it is already added to the not visited list, continue.
      if (!(NotVisited.insert(F)).second)
        continue;

      // Should only process function with a body.
      if (F->isDeclaration())
        continue;

      // Add successors to the stack for future processing.
      for (CallGraphNode::iterator CI = CGN->begin(), CE = CGN->end();
           CI != CE; ++CI)
        stack.push_back(CI->second);
    }
  }
}

/// PropagateCallFrequency - Calculate function call and invocation frequencies.
void StaticProfilePass::PropagateCallFrequency(CallGraphNode *root, bool end) {
  Function *head = root->getFunction();

  std::vector<CallGraphNode *> stack;
  stack.push_back(root);

  do {
    CallGraphNode *node = stack.back();
    stack.pop_back();

    Function *F = node->getFunction();
    DEBUG(errs() << "PropagateCallFrequency: " << F->getName() << ", "
                 << head->getName() << ", " << (end ? "true" : "false")
                 << ")\n");

    // Check if already visited function.
    if (!NotVisited.count(F))
      continue;

    // Run over all predecessors of this function.
    DenseSet<const Function *> &preds = Predecessors[F];
    bool InvalidEdge = false;
    for (DenseSet<const Function *>::iterator PI = preds.begin(),
        PE = preds.end(); PI != PE; ++PI) {
      const Function *predecessor = *PI;
      FunctionEdge fedge = std::make_pair(predecessor, F);

      // Check if we have calculated all predecessors edge previously.
      if (NotVisited.count(predecessor) && !FunctionBackEdges.count(fedge)) {
        InvalidEdge = true;
        break;
      }
    }

    // There is an unprocessed predecessor edge.
    if (InvalidEdge)
      continue;

    // Calculate all incoming edges frequencies and cyclic_frequency for
    // loops.
    double cfreq = (F == head) ? 1.0 : 0.0;
    double cyclic_frequency = 0.0;
    for (DenseSet<const Function *>::iterator PI = preds.begin(),
         PE = preds.end(); PI != PE; ++PI) {
      const Function *predecessor = *PI;
      FunctionEdge fedge = std::make_pair(predecessor, F);

      // Is the edge a back edge.
      bool backedge = FunctionBackEdges.count(fedge);

      // Consider the cyclic_frequency only in the last call to propagate
      // frequency.
      if (end && backedge)
        cyclic_frequency += getBackEdgeFrequency(fedge);
      else if (!backedge)
        cfreq += getGlobalEdgeFrequency(fedge);
    }

    // For loops that seems not to terminate, the cyclic frequency can be
    // higher than 1.0. In this case, limit the cyclic frequency below 1.0.
    if (cyclic_frequency > (1.0 - epsilon))
      cyclic_frequency = 1.0 - epsilon;

    // Calculate invocation frequency.
    cfreq = cfreq / (1.0 - cyclic_frequency);
    FunctionInformation[F] = cfreq;
    DEBUG(errs() << "  Call Frequency[" << F->getName() << "]:  "
                 << format("%.3f", cfreq) << "\n");

    // Mark the function as visited.
    NotVisited.erase(F);

    // Do not process successors for function without a body.
    if (F->isDeclaration())
      continue;

    // Calculate global function edge invocation frequency for successors.
    for (CallGraphNode::iterator FI = node->begin(), FE = node->end();
         FI != FE; ++FI) {
      CallGraphNode *succ_node = FI->second;
      if (Function *successor = succ_node->getFunction()) {
        FunctionEdge fedge = std::make_pair(F, successor);

        // Calculate the global frequency for this function edge.
        double gfreq = LocalEdgeFrequency[fedge] * cfreq;
        GlobalEdgeFrequency[fedge] = gfreq;

        // Print the global edge frequency.
        DEBUG(errs() << "  GlobalEdgeFrequency[(" << F->getName() << "->"
                     << successor->getName() << ")] = " << format("%.3f", gfreq)
                     << "\n");

        // Update back edge frequency in case of loop.
        if (!end && successor == head)
          BackEdgeFrequency[fedge] = gfreq;
      }
    }

    // Call propagate call frequency for function edges that are not back edges.
    std::vector<CallGraphNode *> backedges;
    for (CallGraphNode::iterator FI = node->begin(), FE = node->end();
         FI != FE; ++FI) {
      CallGraphNode *succ_node = FI->second;
      if (Function *successor = succ_node->getFunction()) {
        // Check if it is a back edge.
        if (!FunctionBackEdges.count(std::make_pair(F, successor)))
          backedges.push_back(succ_node);
      }
    }

    // This was done just to ensure that the algorithm would process the
    // left-most child before, in order to simulate normal PropagateCallFreq
    // recursive calls.
    std::vector<CallGraphNode *>::reverse_iterator RI, RE;
    for (RI = backedges.rbegin(), RE = backedges.rend(); RI != RE; ++RI)
      stack.push_back(*RI);
  } while (!stack.empty());
}

/// CalculateGlobalInfo - With calculated function frequency, recalculate block
/// and edge frequencies taking it into consideration.
void StaticProfilePass::CalculateGlobalInfo(Module &M) {

	  for (Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI) {
    Function *F = MI;





    // Obtain call frequency.
    double cfreq = FunctionInformation[F];

    DEBUG(errs() << "Global edges for function: " << F->getName() << "\n");

    // Update edge frequency considering function call frequency.
    EdgeWeights &edges = EdgeInformation[F];
    for (EdgeWeights::iterator EI = edges.begin(), EE = edges.end();
         EI != EE; ++EI)
		 {
			EI->second *= cfreq;
		 }
    DEBUG(errs() << "Global blocks for function: " << F->getName() << "\n");

    // Update block frequency considering function call frequency.
    BlockCounts &blocks = BlockInformation[F];
    for (BlockCounts::iterator BI = blocks.begin(), BE = blocks.end();
         BI != BE; ++BI)
      BI->second *= cfreq;
  }
}

/// getBackEdgeFrequency - Get updated back edges frequency. In case of not
/// found, use the local edge frequency.
double StaticProfilePass::getBackEdgeFrequency(FunctionEdge &fedge) const {
  DenseMap<FunctionEdge, double>::const_iterator I =
      BackEdgeFrequency.find(fedge);
  if (I != BackEdgeFrequency.end())
    return I->second;

  I = LocalEdgeFrequency.find(fedge);
  return (I != LocalEdgeFrequency.end() ? I->second : 0.0);
}

