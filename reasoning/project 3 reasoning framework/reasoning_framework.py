"""
Multi-Agent Reasoning Framework for Protein Engineering with ΔΔG Prediction
Implements strategic mutation selection, stability analysis, and iterative optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import heapq
from collections import defaultdict
import copy


# ============= Core Reasoning Components =============

class ReasoningStrategy(Enum):
    """Different reasoning strategies for mutation selection"""
    GREEDY = "greedy"  # Select best single mutation
    BEAM_SEARCH = "beam_search"  # Keep top-k candidates
    MONTE_CARLO = "monte_carlo"  # Stochastic exploration
    EVOLUTIONARY = "evolutionary"  # Genetic algorithm approach
    ENSEMBLE = "ensemble"  # Combine multiple strategies


@dataclass
class MutationCandidate:
    """Represents a mutation candidate with reasoning trace"""
    position: int
    wild_type: str
    mutant: str
    predicted_ddg: float
    confidence: float
    reasoning_chain: List[str]
    structural_context: Dict
    score: float = 0.0
    
    def __lt__(self, other):
        return self.score < other.score


@dataclass
class ReasoningStep:
    """Single step in reasoning process"""
    step_type: str  # "analyze", "predict", "evaluate", "select"
    input_state: Dict
    output_state: Dict
    reasoning: str
    confidence: float


class ContextAnalyzer:
    """Analyzes structural and functional context around mutation sites"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def analyze_local_environment(self, sequence: str, position: int, 
                                  structure_graph=None) -> Dict:
        """Analyze the local environment around a mutation site"""
        context = {
            'position': position,
            'wild_type': sequence[position] if position < len(sequence) else 'X',
            'sequence_context': self._get_sequence_context(sequence, position),
            'structural_context': {},
            'functional_motifs': [],
            'conservation_score': 0.0
        }
        
        # Analyze sequence context (nearby residues)
        window = 5
        start = max(0, position - window)
        end = min(len(sequence), position + window + 1)
        local_seq = sequence[start:end]
        
        # Check for known functional motifs
        context['functional_motifs'] = self._identify_motifs(local_seq, position - start)
        
        # Structural context (if structure available)
        if structure_graph is not None:
            context['structural_context'] = self._analyze_structure_context(
                structure_graph, position
            )
        
        # Predicted conservation (simple heuristic - can be enhanced)
        context['conservation_score'] = self._estimate_conservation(sequence, position)
        
        return context
    
    def _get_sequence_context(self, sequence: str, position: int, window: int = 5) -> str:
        """Get surrounding sequence context"""
        start = max(0, position - window)
        end = min(len(sequence), position + window + 1)
        return sequence[start:end]
    
    def _identify_motifs(self, local_seq: str, rel_pos: int) -> List[str]:
        """Identify functional motifs (simplified)"""
        motifs = []
        
        # Check for common motifs
        if 'RGD' in local_seq:
            motifs.append('cell_adhesion_motif')
        if 'NXS' in local_seq or 'NXT' in local_seq:
            motifs.append('n_glycosylation_site')
        if local_seq.count('C') >= 2:
            motifs.append('potential_disulfide_bond')
        
        # Check if position is in active site (heuristic: many charged residues)
        charged = sum(1 for aa in local_seq if aa in 'DEKR')
        if charged >= 3:
            motifs.append('potential_active_site')
        
        return motifs
    
    def _analyze_structure_context(self, structure_graph, position: int) -> Dict:
        """Analyze 3D structural context"""
        # Extract structural features
        context = {
            'secondary_structure': 'unknown',
            'solvent_accessibility': 0.5,
            'nearby_residues': [],
            'structural_constraints': []
        }
        
        # This would use actual structural data in practice
        # For now, placeholder logic
        
        return context
    
    def _estimate_conservation(self, sequence: str, position: int) -> float:
        """Estimate conservation score (0-1, higher = more conserved)"""
        # Simplified: hydrophobic core positions are more conserved
        aa = sequence[position]
        hydrophobic = 'AILMFWV'
        charged = 'DEKR'
        
        # Heuristic based on amino acid type and position
        if aa in hydrophobic:
            return 0.7  # Hydrophobic residues often conserved in core
        elif aa in charged:
            return 0.4  # Charged residues more variable
        else:
            return 0.5
    
    def should_avoid_mutation(self, context: Dict) -> Tuple[bool, str]:
        """Determine if position should be avoided for mutation"""
        reasons = []
        
        # Avoid highly conserved positions
        if context['conservation_score'] > 0.8:
            reasons.append("highly conserved position")
        
        # Avoid functional motifs
        if context['functional_motifs']:
            reasons.append(f"part of functional motif: {context['functional_motifs']}")
        
        # Avoid disulfide bonds
        if context['wild_type'] == 'C':
            reasons.append("cysteine potentially involved in disulfide bond")
        
        should_avoid = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else ""
        
        return should_avoid, reason


class DDGReasoner:
    """Core reasoning engine that uses the ΔΔG model"""
    
    def __init__(self, ddg_model, context_analyzer, device='cpu'):
        self.model = ddg_model
        self.context_analyzer = context_analyzer
        self.device = device
        self.model.eval()
        
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
    def predict_single_mutation(self, sequence: str, position: int, 
                               wt_aa: str, mut_aa: str,
                               structure_graph=None) -> Tuple[float, float]:
        """Predict ΔΔG for a single mutation with confidence"""
        with torch.no_grad():
            # Prepare inputs (simplified - adapt to your model's input format)
            seq_tokens = self._tokenize_sequence(sequence)
            mutation_info = torch.tensor([[position / len(sequence), 
                                          self._aa_to_idx(wt_aa),
                                          self._aa_to_idx(mut_aa)]], dtype=torch.float)
            mutation_mask = self._create_mutation_mask(len(sequence), position)
            
            # Move to device
            seq_tokens = seq_tokens.to(self.device)
            mutation_info = mutation_info.to(self.device)
            mutation_mask = mutation_mask.to(self.device)
            
            # Predict
            if structure_graph is not None:
                structure_graph = structure_graph.to(self.device)
                ddg_pred = self.model(seq_tokens, structure_graph, 
                                     mutation_info, mutation_mask)
            else:
                # If no structure, use sequence-only mode (if model supports)
                ddg_pred = self.model(seq_tokens, None, mutation_info, mutation_mask)
            
            ddg_value = ddg_pred.item()
            
            # Estimate confidence based on model features
            confidence = self._estimate_confidence(ddg_value, sequence, position)
            
        return ddg_value, confidence
    
    def _tokenize_sequence(self, sequence: str) -> torch.Tensor:
        """Convert sequence to token tensor"""
        from collections import OrderedDict
        vocab = ['<PAD>', '<MASK>', '<CLS>', '<SEP>'] + self.amino_acids
        aa_to_idx = {aa: i for i, aa in enumerate(vocab)}
        tokens = [aa_to_idx.get(aa, 0) for aa in sequence]
        return torch.tensor([tokens], dtype=torch.long)
    
    def _aa_to_idx(self, aa: str) -> int:
        """Convert amino acid to index"""
        vocab = ['<PAD>', '<MASK>', '<CLS>', '<SEP>'] + self.amino_acids
        aa_to_idx = {aa: i for i, aa in enumerate(vocab)}
        return aa_to_idx.get(aa, 0)
    
    def _create_mutation_mask(self, seq_len: int, position: int) -> torch.Tensor:
        """Create binary mask for mutation position"""
        mask = torch.zeros(1, seq_len)
        if 0 <= position < seq_len:
            mask[0, position] = 1
        return mask
    
    def _estimate_confidence(self, ddg_value: float, sequence: str, 
                           position: int) -> float:
        """Estimate prediction confidence"""
        # Higher confidence for moderate ΔΔG values
        # Lower confidence for extreme values or unusual positions
        
        confidence = 1.0
        
        # Penalize extreme predictions
        if abs(ddg_value) > 5.0:
            confidence *= 0.7
        
        # Penalize if mutation is at sequence termini
        if position < 5 or position > len(sequence) - 5:
            confidence *= 0.8
        
        # Adjust based on context
        context = self.context_analyzer.analyze_local_environment(sequence, position)
        if context['functional_motifs']:
            confidence *= 0.9
        
        return max(0.1, min(1.0, confidence))
    
    def reason_about_mutation(self, sequence: str, position: int,
                            wt_aa: str, mut_aa: str,
                            structure_graph=None) -> Dict:
        """Generate reasoning chain for a mutation"""
        reasoning_chain = []
        
        # Step 1: Context analysis
        context = self.context_analyzer.analyze_local_environment(
            sequence, position, structure_graph
        )
        reasoning_chain.append(f"Analyzing position {position} ({wt_aa})")
        reasoning_chain.append(f"Local context: {context['sequence_context']}")
        
        # Step 2: Check constraints
        should_avoid, reason = self.context_analyzer.should_avoid_mutation(context)
        if should_avoid:
            reasoning_chain.append(f"⚠️ Caution: {reason}")
        
        # Step 3: Predict ΔΔG
        ddg, confidence = self.predict_single_mutation(
            sequence, position, wt_aa, mut_aa, structure_graph
        )
        reasoning_chain.append(f"Predicted ΔΔG: {ddg:.2f} kcal/mol (confidence: {confidence:.2f})")
        
        # Step 4: Interpret result
        interpretation = self._interpret_ddg(ddg, wt_aa, mut_aa)
        reasoning_chain.append(interpretation)
        
        return {
            'position': position,
            'mutation': f"{wt_aa}{position}{mut_aa}",
            'ddg': ddg,
            'confidence': confidence,
            'context': context,
            'should_avoid': should_avoid,
            'reasoning_chain': reasoning_chain,
            'interpretation': interpretation
        }
    
    def _interpret_ddg(self, ddg: float, wt_aa: str, mut_aa: str) -> str:
        """Interpret ΔΔG value"""
        if ddg < -2.0:
            return f"✅ Strongly stabilizing: {wt_aa}→{mut_aa} significantly increases stability"
        elif ddg < -0.5:
            return f"✓ Stabilizing: {wt_aa}→{mut_aa} improves stability"
        elif ddg < 0.5:
            return f"→ Neutral: {wt_aa}→{mut_aa} has minimal effect on stability"
        elif ddg < 2.0:
            return f"⚠️ Destabilizing: {wt_aa}→{mut_aa} reduces stability"
        else:
            return f"❌ Strongly destabilizing: {wt_aa}→{mut_aa} significantly reduces stability"


# ============= Strategic Search Algorithms =============

class MutationSearchEngine:
    """Search for optimal mutations using various strategies"""
    
    def __init__(self, reasoner: DDGReasoner):
        self.reasoner = reasoner
        
    def greedy_search(self, sequence: str, structure_graph=None,
                     goal: str = "stabilize", max_mutations: int = 5) -> List[MutationCandidate]:
        """Greedy search: iteratively select best single mutation"""
        print("🔍 Starting greedy search...")
        current_seq = sequence
        selected_mutations = []
        reasoning_log = []
        
        for iteration in range(max_mutations):
            print(f"\n  Iteration {iteration + 1}/{max_mutations}")
            best_candidate = None
            best_score = float('-inf') if goal == "stabilize" else float('inf')
            
            # Try all possible single mutations
            candidates = self._generate_single_mutation_candidates(current_seq)
            print(f"    Evaluating {len(candidates)} candidates...")
            
            for pos, wt_aa, mut_aa in candidates:
                result = self.reasoner.reason_about_mutation(
                    current_seq, pos, wt_aa, mut_aa, structure_graph
                )
                
                score = self._compute_score(result, goal)
                
                if (goal == "stabilize" and score > best_score) or \
                   (goal == "destabilize" and score < best_score):
                    best_score = score
                    best_candidate = MutationCandidate(
                        position=pos,
                        wild_type=wt_aa,
                        mutant=mut_aa,
                        predicted_ddg=result['ddg'],
                        confidence=result['confidence'],
                        reasoning_chain=result['reasoning_chain'],
                        structural_context=result['context'],
                        score=score
                    )
            
            if best_candidate is None:
                print("    No beneficial mutations found.")
                break
            
            # Apply mutation
            current_seq = self._apply_mutation(current_seq, best_candidate)
            selected_mutations.append(best_candidate)
            
            print(f"    ✓ Selected: {best_candidate.wild_type}{best_candidate.position}{best_candidate.mutant}")
            print(f"      ΔΔG: {best_candidate.predicted_ddg:.2f}, Score: {best_candidate.score:.3f}")
        
        return selected_mutations
    
    def beam_search(self, sequence: str, structure_graph=None,
                   goal: str = "stabilize", beam_width: int = 3,
                   max_depth: int = 3) -> List[List[MutationCandidate]]:
        """Beam search: maintain top-k mutation sequences"""
        print(f"🔍 Starting beam search (width={beam_width}, depth={max_depth})...")
        
        # Initialize beam with empty mutation sequence
        beam = [([], sequence, 0.0)]  # (mutations, sequence, cumulative_score)
        
        for depth in range(max_depth):
            print(f"\n  Depth {depth + 1}/{max_depth}")
            candidates = []
            
            # Expand each sequence in beam
            for mutations, seq, cum_score in beam:
                single_muts = self._generate_single_mutation_candidates(seq, limit=50)
                
                for pos, wt_aa, mut_aa in single_muts:
                    result = self.reasoner.reason_about_mutation(
                        seq, pos, wt_aa, mut_aa, structure_graph
                    )
                    
                    score = self._compute_score(result, goal)
                    new_cum_score = cum_score + score
                    
                    candidate = MutationCandidate(
                        position=pos, wild_type=wt_aa, mutant=mut_aa,
                        predicted_ddg=result['ddg'],
                        confidence=result['confidence'],
                        reasoning_chain=result['reasoning_chain'],
                        structural_context=result['context'],
                        score=score
                    )
                    
                    new_seq = self._apply_mutation(seq, candidate)
                    new_mutations = mutations + [candidate]
                    
                    candidates.append((new_mutations, new_seq, new_cum_score))
            
            # Keep top beam_width candidates
            if goal == "stabilize":
                candidates.sort(key=lambda x: x[2], reverse=True)
            else:
                candidates.sort(key=lambda x: x[2])
            
            beam = candidates[:beam_width]
            
            print(f"    Top {len(beam)} sequences:")
            for i, (muts, _, score) in enumerate(beam[:3]):
                mut_str = ", ".join([f"{m.wild_type}{m.position}{m.mutant}" for m in muts])
                print(f"      {i+1}. [{mut_str}] Score: {score:.3f}")
        
        return [mutations for mutations, _, _ in beam]
    
    def monte_carlo_search(self, sequence: str, structure_graph=None,
                          goal: str = "stabilize", n_iterations: int = 100,
                          temperature: float = 1.0) -> List[MutationCandidate]:
        """Monte Carlo search with simulated annealing"""
        print(f"🔍 Starting Monte Carlo search ({n_iterations} iterations)...")
        
        current_seq = sequence
        current_mutations = []
        current_score = 0.0
        
        best_seq = sequence
        best_mutations = []
        best_score = float('-inf') if goal == "stabilize" else float('inf')
        
        for iteration in range(n_iterations):
            # Propose random mutation
            candidates = self._generate_single_mutation_candidates(current_seq, limit=20)
            if not candidates:
                break
            
            pos, wt_aa, mut_aa = candidates[np.random.randint(len(candidates))]
            
            result = self.reasoner.reason_about_mutation(
                current_seq, pos, wt_aa, mut_aa, structure_graph
            )
            
            proposed_score = current_score + self._compute_score(result, goal)
            
            # Accept or reject (Metropolis criterion)
            delta = proposed_score - current_score
            if goal == "destabilize":
                delta = -delta
            
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                # Accept
                candidate = MutationCandidate(
                    position=pos, wild_type=wt_aa, mutant=mut_aa,
                    predicted_ddg=result['ddg'],
                    confidence=result['confidence'],
                    reasoning_chain=result['reasoning_chain'],
                    structural_context=result['context'],
                    score=self._compute_score(result, goal)
                )
                
                current_seq = self._apply_mutation(current_seq, candidate)
                current_mutations.append(candidate)
                current_score = proposed_score
                
                # Update best
                if (goal == "stabilize" and current_score > best_score) or \
                   (goal == "destabilize" and current_score < best_score):
                    best_score = current_score
                    best_mutations = current_mutations.copy()
                    best_seq = current_seq
            
            # Cool down temperature
            temperature *= 0.995
            
            if (iteration + 1) % 20 == 0:
                print(f"    Iteration {iteration + 1}: Best score = {best_score:.3f}, "
                      f"Current mutations = {len(current_mutations)}")
        
        return best_mutations
    
    def _generate_single_mutation_candidates(self, sequence: str, 
                                            limit: Optional[int] = None) -> List[Tuple]:
        """Generate all possible single point mutations"""
        candidates = []
        amino_acids = self.reasoner.amino_acids
        
        for pos, wt_aa in enumerate(sequence):
            for mut_aa in amino_acids:
                if mut_aa != wt_aa:
                    candidates.append((pos, wt_aa, mut_aa))
        
        if limit:
            candidates = candidates[:limit]
        
        return candidates
    
    def _compute_score(self, result: Dict, goal: str) -> float:
        """Compute mutation score based on goal"""
        ddg = result['ddg']
        confidence = result['confidence']
        should_avoid = result['should_avoid']
        
        # Base score from ΔΔG
        if goal == "stabilize":
            score = -ddg  # More negative ΔΔG = better
        elif goal == "destabilize":
            score = ddg  # More positive ΔΔG = better
        else:
            score = -abs(ddg)  # Neutral mutations
        
        # Weight by confidence
        score *= confidence
        
        # Penalty for positions that should be avoided
        if should_avoid:
            score *= 0.5
        
        return score
    
    def _apply_mutation(self, sequence: str, mutation: MutationCandidate) -> str:
        """Apply mutation to sequence"""
        seq_list = list(sequence)
        seq_list[mutation.position] = mutation.mutant
        return ''.join(seq_list)


# ============= High-Level Reasoning Interface =============

class ProteinEngineeringAgent:
    """High-level agent for protein engineering tasks"""
    
    def __init__(self, ddg_model, device='cpu'):
        context_analyzer = ContextAnalyzer(ddg_model, device)
        reasoner = DDGReasoner(ddg_model, context_analyzer, device)
        self.search_engine = MutationSearchEngine(reasoner)
        self.reasoner = reasoner
        
    def optimize_stability(self, sequence: str, structure_graph=None,
                          strategy: ReasoningStrategy = ReasoningStrategy.GREEDY,
                          **kwargs) -> Dict:
        """Optimize protein stability"""
        print("🎯 Goal: Optimize protein stability")
        print(f"   Strategy: {strategy.value}")
        print(f"   Sequence length: {len(sequence)}")
        print()
        
        if strategy == ReasoningStrategy.GREEDY:
            mutations = self.search_engine.greedy_search(
                sequence, structure_graph, goal="stabilize", **kwargs
            )
        elif strategy == ReasoningStrategy.BEAM_SEARCH:
            mutation_seqs = self.search_engine.beam_search(
                sequence, structure_graph, goal="stabilize", **kwargs
            )
            mutations = mutation_seqs[0] if mutation_seqs else []
        elif strategy == ReasoningStrategy.MONTE_CARLO:
            mutations = self.search_engine.monte_carlo_search(
                sequence, structure_graph, goal="stabilize", **kwargs
            )
        else:
            raise ValueError(f"Strategy {strategy} not implemented")
        
        return self._summarize_results(sequence, mutations, "stabilize")
    
    def design_destabilizing_mutations(self, sequence: str, structure_graph=None,
                                      strategy: ReasoningStrategy = ReasoningStrategy.GREEDY,
                                      **kwargs) -> Dict:
        """Design destabilizing mutations (e.g., for controlled degradation)"""
        print("🎯 Goal: Design destabilizing mutations")
        print(f"   Strategy: {strategy.value}")
        print()
        
        if strategy == ReasoningStrategy.GREEDY:
            mutations = self.search_engine.greedy_search(
                sequence, structure_graph, goal="destabilize", **kwargs
            )
        else:
            raise ValueError(f"Strategy {strategy} not implemented for destabilization")
        
        return self._summarize_results(sequence, mutations, "destabilize")
    
    def analyze_mutation_set(self, sequence: str, mutations: List[Tuple[int, str, str]],
                           structure_graph=None) -> Dict:
        """Analyze a specific set of mutations"""
        print("🔬 Analyzing mutation set...")
        results = []
        
        for pos, wt_aa, mut_aa in mutations:
            result = self.reasoner.reason_about_mutation(
                sequence, pos, wt_aa, mut_aa, structure_graph
            )
            results.append(result)
            
            print(f"\n   {wt_aa}{pos}{mut_aa}:")
            print(f"     ΔΔG: {result['ddg']:.2f} kcal/mol")
            print(f"     {result['interpretation']}")
        
        return {'individual_results': results}
    
    def _summarize_results(self, original_seq: str, mutations: List[MutationCandidate],
                          goal: str) -> Dict:
        """Summarize optimization results"""
        print("\n" + "="*60)
        print("📊 OPTIMIZATION SUMMARY")
        print("="*60)
        
        if not mutations:
            print("No beneficial mutations found.")
            return {'mutations': [], 'final_sequence': original_seq}
        
        # Apply all mutations
        final_seq = original_seq
        total_ddg = 0.0
        
        print(f"\n{len(mutations)} mutations selected:")
        for i, mut in enumerate(mutations, 1):
            print(f"\n{i}. {mut.wild_type}{mut.position}{mut.mutant}")
            print(f"   ΔΔG: {mut.predicted_ddg:.2f} kcal/mol (confidence: {mut.confidence:.2f})")
            print(f"   Reasoning: {mut.reasoning_chain[-1]}")
            
            final_seq = self.search_engine._apply_mutation(final_seq, mut)
            total_ddg += mut.predicted_ddg
        
        print(f"\nTotal predicted ΔΔG change: {total_ddg:.2f} kcal/mol")
        print(f"Final sequence: {final_seq[:50]}..." if len(final_seq) > 50 else f"Final sequence: {final_seq}")
        
        return {
            'mutations': mutations,
            'original_sequence': original_seq,
            'final_sequence': final_seq,
            'total_ddg': total_ddg,
            'goal': goal
        }


# ============= Example Usage =============

if __name__ == "__main__":
    print("🧬 Protein Engineering Reasoning Framework")
    print("="*60)
    
    # Mock model for demonstration (replace with actual trained model)
    class MockDDGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, seq_tokens, structure_graph, mutation_info, mutation_mask):
            # Mock prediction
            return torch.randn(seq_tokens.size(0), 1) * 2
    
    mock_model = MockDDGModel()
    
    # Initialize agent
    agent = ProteinEngineeringAgent(mock_model, device='cpu')
    
    # Example sequence (small protein)
    example_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLS"
    
    print(f"\nOriginal sequence: {example_seq}")
    print(f"Length: {len(example_seq)} residues")
    
    # Scenario 1: Optimize stability with greedy search
    print("\n" + "="*60)
    print("SCENARIO 1: Stability Optimization (Greedy)")
    print("="*60)
    results = agent.optimize_stability(
        example_seq,
        strategy=ReasoningStrategy.GREEDY,
        max_mutations=3
    )
    
    # Scenario 2: Analyze specific mutations
    print("\n" + "="*60)
    print("SCENARIO 2: Analyze Specific Mutations")
    print("="*60)
    specific_mutations = [(10, 'Q', 'A'), (20, 'E', 'K')]
    agent.analyze_mutation_set(example_seq, specific_mutations)
    
    print("\n" + "="*60)
    print("Framework Ready!")
    print("="*60)
    print("\nAvailable strategies:")
    print("  - GREEDY: Fast, locally optimal")
    print("  - BEAM_SEARCH: Explores multiple paths")
    print("  - MONTE_CARLO: Stochastic global search")
    print("\nUse with your trained ΔΔG model for real protein engineering!")


# ============= Advanced Reasoning Features =============

class EnsembleReasoner:
    """Combine multiple reasoning strategies for robust predictions"""
    
    def __init__(self, agent: ProteinEngineeringAgent):
        self.agent = agent
        
    def ensemble_optimize(self, sequence: str, structure_graph=None,
                         strategies: List[ReasoningStrategy] = None,
                         voting: str = "weighted") -> Dict:
        """Run multiple strategies and combine results"""
        if strategies is None:
            strategies = [
                ReasoningStrategy.GREEDY,
                ReasoningStrategy.BEAM_SEARCH,
                ReasoningStrategy.MONTE_CARLO
            ]
        
        print("🔮 Ensemble Optimization")
        print(f"   Running {len(strategies)} strategies...")
        
        all_mutations = defaultdict(list)  # mutation -> list of (ddg, confidence, strategy)
        
        for strategy in strategies:
            print(f"\n   → Running {strategy.value}...")
            if strategy == ReasoningStrategy.GREEDY:
                results = self.agent.search_engine.greedy_search(
                    sequence, structure_graph, max_mutations=3
                )
            elif strategy == ReasoningStrategy.BEAM_SEARCH:
                results_list = self.agent.search_engine.beam_search(
                    sequence, structure_graph, beam_width=2, max_depth=2
                )
                results = results_list[0] if results_list else []
            elif strategy == ReasoningStrategy.MONTE_CARLO:
                results = self.agent.search_engine.monte_carlo_search(
                    sequence, structure_graph, n_iterations=50
                )
            else:
                continue
            
            # Collect mutations
            for mut in results:
                key = (mut.position, mut.wild_type, mut.mutant)
                all_mutations[key].append((mut.predicted_ddg, mut.confidence, strategy.value))
        
        # Vote on best mutations
        final_mutations = self._vote_on_mutations(all_mutations, voting)
        
        print("\n📊 Ensemble Results:")
        for mut_key, vote_info in final_mutations.items():
            pos, wt, mt = mut_key
            print(f"   {wt}{pos}{mt}: {vote_info['strategies']} strategies, "
                  f"avg ΔΔG={vote_info['avg_ddg']:.2f}")
        
        return {
            'consensus_mutations': final_mutations,
            'strategy_results': all_mutations
        }
    
    def _vote_on_mutations(self, all_mutations: Dict, voting: str) -> Dict:
        """Aggregate mutations across strategies"""
        scored_mutations = {}
        
        for mut_key, predictions in all_mutations.items():
            n_strategies = len(predictions)
            avg_ddg = np.mean([p[0] for p in predictions])
            avg_confidence = np.mean([p[1] for p in predictions])
            strategies = [p[2] for p in predictions]
            
            if voting == "weighted":
                # Weight by confidence and number of strategies
                score = -avg_ddg * avg_confidence * (n_strategies / 3.0)
            elif voting == "majority":
                # Require at least 2/3 strategies to agree
                score = n_strategies if n_strategies >= 2 else 0
            else:
                score = -avg_ddg
            
            scored_mutations[mut_key] = {
                'score': score,
                'avg_ddg': avg_ddg,
                'avg_confidence': avg_confidence,
                'strategies': strategies,
                'n_votes': n_strategies
            }
        
        # Return top mutations
        sorted_muts = sorted(scored_mutations.items(), key=lambda x: x[1]['score'], reverse=True)
        return dict(sorted_muts[:5])


class ChainOfThoughtReasoner:
    """Generate detailed reasoning chains for mutation decisions"""
    
    def __init__(self, reasoner: DDGReasoner):
        self.reasoner = reasoner
        
    def explain_mutation_decision(self, sequence: str, position: int,
                                  wt_aa: str, mut_aa: str,
                                  structure_graph=None) -> str:
        """Generate human-readable explanation"""
        result = self.reasoner.reason_about_mutation(
            sequence, position, wt_aa, mut_aa, structure_graph
        )
        
        explanation = []
        explanation.append(f"🔬 Mutation Analysis: {wt_aa}{position}→{mut_aa}")
        explanation.append("\n" + "="*50)
        
        # Step 1: Context
        explanation.append("\n1️⃣ CONTEXT ANALYSIS")
        context = result['context']
        explanation.append(f"   Position: {position}/{len(sequence)}")
        explanation.append(f"   Local sequence: ...{context['sequence_context']}...")
        
        if context['functional_motifs']:
            explanation.append(f"   ⚠️ Functional motifs: {', '.join(context['functional_motifs'])}")
        else:
            explanation.append("   ✓ No critical motifs detected")
        
        explanation.append(f"   Conservation: {context['conservation_score']:.2f}")
        
        # Step 2: Prediction
        explanation.append("\n2️⃣ PREDICTION")
        explanation.append(f"   ΔΔG = {result['ddg']:.2f} kcal/mol")
        explanation.append(f"   Confidence = {result['confidence']:.2f}")
        
        # Step 3: Interpretation
        explanation.append("\n3️⃣ INTERPRETATION")
        explanation.append(f"   {result['interpretation']}")
        
        # Step 4: Recommendation
        explanation.append("\n4️⃣ RECOMMENDATION")
        if result['should_avoid']:
            explanation.append("   ❌ NOT RECOMMENDED")
            explanation.append(f"   Reason: {result['reasoning_chain'][1]}")
        elif result['ddg'] < -1.0:
            explanation.append("   ✅ HIGHLY RECOMMENDED")
            explanation.append("   Expected to significantly improve stability")
        elif result['ddg'] < 0:
            explanation.append("   ✓ RECOMMENDED")
            explanation.append("   Expected to moderately improve stability")
        else:
            explanation.append("   ⚠️ CAUTION")
            explanation.append("   May reduce protein stability")
        
        # Step 5: Molecular reasoning
        explanation.append("\n5️⃣ MOLECULAR RATIONALE")
        explanation.append(self._generate_molecular_reasoning(wt_aa, mut_aa, result))
        
        return "\n".join(explanation)
    
    def _generate_molecular_reasoning(self, wt_aa: str, mut_aa: str, result: Dict) -> str:
        """Generate molecular-level explanation"""
        reasons = []
        
        # Hydrophobicity
        hydrophobic = set('AILMFWV')
        hydrophilic = set('STNQ')
        charged = set('DEKR')
        
        wt_hydro = wt_aa in hydrophobic
        mut_hydro = mut_aa in hydrophobic
        
        if wt_hydro and not mut_hydro:
            reasons.append("   • Replacing hydrophobic residue with more polar residue")
            reasons.append("     → May affect core packing or surface exposure")
        elif not wt_hydro and mut_hydro:
            reasons.append("   • Introducing hydrophobic residue")
            reasons.append("     → May improve core stability if buried")
        
        # Charge
        wt_charged = wt_aa in charged
        mut_charged = mut_aa in charged
        
        if wt_charged and not mut_charged:
            reasons.append("   • Removing charged residue")
            reasons.append("     → May affect electrostatic interactions or pH stability")
        elif not wt_charged and mut_charged:
            reasons.append("   • Introducing charged residue")
            reasons.append("     → May form new salt bridges or affect solubility")
        
        # Size
        small = set('AGCS')
        large = set('WYFR')
        
        if wt_aa in small and mut_aa in large:
            reasons.append("   • Increasing residue size")
            reasons.append("     → May cause steric clashes or improve packing")
        elif wt_aa in large and mut_aa in small:
            reasons.append("   • Decreasing residue size")
            reasons.append("     → May create cavities or improve flexibility")
        
        # Special cases
        if wt_aa == 'P' or mut_aa == 'P':
            reasons.append("   • Proline involved - affects backbone conformational freedom")
        if wt_aa == 'G' or mut_aa == 'G':
            reasons.append("   • Glycine involved - most flexible residue")
        if wt_aa == 'C' or mut_aa == 'C':
            reasons.append("   • Cysteine involved - potential disulfide bond effects")
        
        if not reasons:
            reasons.append("   • Moderate substitution within similar physicochemical class")
        
        return "\n".join(reasons)


class InteractiveReasoningSession:
    """Interactive session for exploring mutations"""
    
    def __init__(self, agent: ProteinEngineeringAgent):
        self.agent = agent
        self.cot_reasoner = ChainOfThoughtReasoner(agent.reasoner)
        self.ensemble = EnsembleReasoner(agent)
        self.history = []
        
    def start_session(self, sequence: str, structure_graph=None):
        """Start interactive reasoning session"""
        print("🎮 Interactive Reasoning Session Started")
        print("="*60)
        print(f"Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        print(f"Length: {len(sequence)} residues")
        print("\nAvailable commands:")
        print("  1. analyze <pos> <wt> <mut>  - Analyze specific mutation")
        print("  2. optimize <strategy>       - Run optimization")
        print("  3. ensemble                  - Run ensemble analysis")
        print("  4. history                   - Show analysis history")
        print("  5. compare <mut1> <mut2>     - Compare mutations")
        print("  6. help                      - Show this help")
        print("="*60)
        
        self.sequence = sequence
        self.structure_graph = structure_graph
    
    def analyze_mutation(self, position: int, wt_aa: str, mut_aa: str):
        """Analyze a specific mutation with detailed explanation"""
        explanation = self.cot_reasoner.explain_mutation_decision(
            self.sequence, position, wt_aa, mut_aa, self.structure_graph
        )
        print(explanation)
        
        self.history.append({
            'type': 'analyze',
            'mutation': f"{wt_aa}{position}{mut_aa}",
            'timestamp': 'now'
        })
    
    def compare_mutations(self, mutations: List[Tuple[int, str, str]]):
        """Compare multiple mutations side-by-side"""
        print("📊 Mutation Comparison")
        print("="*60)
        
        results = []
        for pos, wt, mt in mutations:
            result = self.agent.reasoner.reason_about_mutation(
                self.sequence, pos, wt, mt, self.structure_graph
            )
            results.append((f"{wt}{pos}{mt}", result))
        
        # Print comparison table
        print(f"\n{'Mutation':<12} {'ΔΔG':<10} {'Confidence':<12} {'Interpretation'}")
        print("-" * 70)
        
        for mut_str, result in results:
            ddg = result['ddg']
            conf = result['confidence']
            interp = "Stabilizing" if ddg < 0 else "Destabilizing"
            print(f"{mut_str:<12} {ddg:<10.2f} {conf:<12.2f} {interp}")
        
        # Recommendation
        print("\n🎯 Recommendation:")
        best_result = min(results, key=lambda x: x[1]['ddg'])
        print(f"   Best option: {best_result[0]} (ΔΔG = {best_result[1]['ddg']:.2f})")


class ConstraintSatisfactionReasoner:
    """Reason about mutations with multiple constraints"""
    
    def __init__(self, reasoner: DDGReasoner):
        self.reasoner = reasoner
        
    def find_constrained_mutations(self, sequence: str, 
                                   constraints: Dict,
                                   structure_graph=None) -> List[MutationCandidate]:
        """Find mutations satisfying multiple constraints"""
        print("🎯 Constraint-based Mutation Search")
        print("="*60)
        print("Constraints:")
        for key, value in constraints.items():
            print(f"   {key}: {value}")
        print()
        
        candidates = []
        amino_acids = self.reasoner.amino_acids
        
        # Extract constraints
        min_ddg = constraints.get('min_ddg', None)
        max_ddg = constraints.get('max_ddg', None)
        min_confidence = constraints.get('min_confidence', 0.5)
        forbidden_positions = set(constraints.get('forbidden_positions', []))
        required_properties = constraints.get('required_properties', [])
        
        # Search through all possible mutations
        for pos in range(len(sequence)):
            if pos in forbidden_positions:
                continue
            
            wt_aa = sequence[pos]
            
            for mut_aa in amino_acids:
                if mut_aa == wt_aa:
                    continue
                
                # Check if mutation satisfies property requirements
                if required_properties:
                    if not self._check_properties(wt_aa, mut_aa, required_properties):
                        continue
                
                # Predict and check constraints
                result = self.reasoner.reason_about_mutation(
                    sequence, pos, wt_aa, mut_aa, structure_graph
                )
                
                ddg = result['ddg']
                confidence = result['confidence']
                
                # Check ΔΔG constraints
                if min_ddg is not None and ddg < min_ddg:
                    continue
                if max_ddg is not None and ddg > max_ddg:
                    continue
                if confidence < min_confidence:
                    continue
                
                # Add to candidates
                candidate = MutationCandidate(
                    position=pos, wild_type=wt_aa, mutant=mut_aa,
                    predicted_ddg=ddg, confidence=confidence,
                    reasoning_chain=result['reasoning_chain'],
                    structural_context=result['context'],
                    score=-ddg * confidence
                )
                candidates.append(candidate)
        
        # Sort by score
        candidates.sort(reverse=True)
        
        print(f"Found {len(candidates)} mutations satisfying all constraints")
        if candidates:
            print("\nTop 5 candidates:")
            for i, cand in enumerate(candidates[:5], 1):
                print(f"   {i}. {cand.wild_type}{cand.position}{cand.mutant}: "
                      f"ΔΔG={cand.predicted_ddg:.2f}, conf={cand.confidence:.2f}")
        
        return candidates
    
    def _check_properties(self, wt_aa: str, mut_aa: str, 
                         required_properties: List[str]) -> bool:
        """Check if mutation satisfies required physicochemical properties"""
        hydrophobic = set('AILMFWV')
        polar = set('STNQ')
        charged = set('DEKR')
        aromatic = set('FWY')
        
        for prop in required_properties:
            if prop == "increase_hydrophobicity":
                if not (mut_aa in hydrophobic and wt_aa not in hydrophobic):
                    return False
            elif prop == "decrease_hydrophobicity":
                if not (wt_aa in hydrophobic and mut_aa not in hydrophobic):
                    return False
            elif prop == "introduce_charge":
                if mut_aa not in charged:
                    return False
            elif prop == "remove_charge":
                if wt_aa not in charged:
                    return False
            elif prop == "maintain_size":
                small = set('AGCS')
                medium = set('NVDTPIL')
                large = set('MFYWERKQ')
                wt_size = 'S' if wt_aa in small else 'M' if wt_aa in medium else 'L'
                mut_size = 'S' if mut_aa in small else 'M' if mut_aa in medium else 'L'
                if wt_size != mut_size:
                    return False
        
        return True


# ============= Extended Example Usage =============

def demo_advanced_features():
    """Demonstrate advanced reasoning features"""
    print("\n" + "="*60)
    print("🚀 ADVANCED REASONING FEATURES DEMO")
    print("="*60)
    
    # Setup (using mock model)
    class MockDDGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, seq_tokens, structure_graph, mutation_info, mutation_mask):
            return torch.randn(seq_tokens.size(0), 1) * 2
    
    mock_model = MockDDGModel()
    agent = ProteinEngineeringAgent(mock_model, device='cpu')
    
    example_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLS"
    
    # Feature 1: Chain-of-Thought Reasoning
    print("\n" + "="*60)
    print("FEATURE 1: Chain-of-Thought Reasoning")
    print("="*60)
    cot = ChainOfThoughtReasoner(agent.reasoner)
    explanation = cot.explain_mutation_decision(example_seq, 10, 'Q', 'A')
    print(explanation)
    
    # Feature 2: Interactive Session
    print("\n" + "="*60)
    print("FEATURE 2: Interactive Session")
    print("="*60)
    session = InteractiveReasoningSession(agent)
    session.start_session(example_seq)
    session.analyze_mutation(15, 'I', 'V')
    
    # Feature 3: Mutation Comparison
    print("\n" + "="*60)
    print("FEATURE 3: Mutation Comparison")
    print("="*60)
    session.compare_mutations([(10, 'Q', 'A'), (15, 'I', 'V'), (20, 'E', 'D')])
    
    # Feature 4: Constraint Satisfaction
    print("\n" + "="*60)
    print("FEATURE 4: Constraint-Based Search")
    print("="*60)
    constraint_reasoner = ConstraintSatisfactionReasoner(agent.reasoner)
    constraints = {
        'min_ddg': -3.0,
        'max_ddg': -0.5,
        'min_confidence': 0.7,
        'forbidden_positions': [0, 1, 2, len(example_seq)-1],
        'required_properties': ['increase_hydrophobicity']
    }
    candidates = constraint_reasoner.find_constrained_mutations(
        example_seq, constraints
    )
    
    # Feature 5: Ensemble Reasoning
    print("\n" + "="*60)
    print("FEATURE 5: Ensemble Reasoning")
    print("="*60)
    ensemble = EnsembleReasoner(agent)
    ensemble_results = ensemble.ensemble_optimize(example_seq)
    
    print("\n" + "="*60)
    print("✅ All Advanced Features Demonstrated!")
    print("="*60)


if __name__ == "__main__":
    demo_advanced_features()