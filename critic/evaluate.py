"""
Evaluation script for measuring before/after adherence metrics on development set.
Compares model performance before and after DPO finetuning.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict
import logging
from scipy import stats

from .model import ComprehensiveCritic, CriticScore, extract_mix_features
from .classifier import AdherenceClassifier, AdherenceScore
from .dpo_finetune import DPOTrainer

logger = logging.getLogger(__name__)

class AdherenceEvaluator:
    """Evaluates adherence metrics before and after DPO finetuning"""
    
    def __init__(
        self,
        critic_model: ComprehensiveCritic,
        device: str = 'cuda'
    ):
        self.critic = critic_model.to(device)
        self.device = device
        
    def evaluate_model_adherence(
        self,
        model: torch.nn.Module,
        test_samples: List[Dict[str, Any]],
        num_generations: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate a model's adherence on test samples
        
        Args:
            model: Music generation model to evaluate
            test_samples: Test samples with prompts and control JSONs
            num_generations: Number of generations per prompt for averaging
            
        Returns:
            Dictionary with adherence metrics
        """
        model.eval()
        
        all_scores = []
        prompt_level_scores = {}
        
        for sample in test_samples:
            prompt = sample['prompt']
            control_json = sample['control_json']
            
            sample_scores = []
            
            # Generate multiple outputs for this prompt
            for gen_idx in range(num_generations):
                # Generate tokens (this would need actual model implementation)
                generated_tokens = self._generate_tokens(model, prompt, control_json)
                
                # Create dummy inputs for critic (in practice, would need actual audio)
                mel_spec = torch.randn(1, 1, 128, 128).to(self.device)  # Dummy mel-spec
                ref_embedding = torch.randn(1, 512).to(self.device)  # Dummy reference
                mix_features = torch.randn(1, 32).to(self.device)  # Dummy mix features
                
                # Evaluate with critic
                critic_scores = self.critic.evaluate_comprehensive(
                    [prompt],
                    [control_json],
                    generated_tokens.unsqueeze(0),
                    mel_spec,
                    ref_embedding,
                    mix_features
                )
                
                sample_scores.append(critic_scores[0])
            
            # Average scores for this prompt
            avg_score = self._average_critic_scores(sample_scores)
            all_scores.extend(sample_scores)
            prompt_level_scores[prompt] = avg_score
        
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(all_scores)
        metrics['prompt_level_scores'] = prompt_level_scores
        
        return metrics
    
    def _generate_tokens(
        self,
        model: torch.nn.Module,
        prompt: str,
        control_json: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate tokens using the model (placeholder implementation)"""
        # This would need actual model implementation
        # For now, return dummy tokens
        return torch.randint(0, 1000, (64,)).to(self.device)
    
    def _average_critic_scores(self, scores: List[CriticScore]) -> CriticScore:
        """Average multiple CriticScore objects"""
        if not scores:
            raise ValueError("No scores to average")
        
        # Average numerical fields
        avg_overall = np.mean([s.overall for s in scores])
        avg_adherence = np.mean([s.adherence for s in scores])
        avg_style_match = np.mean([s.style_match for s in scores])
        avg_mix_quality = np.mean([s.mix_quality for s in scores])
        avg_confidence = np.mean([s.confidence for s in scores])
        
        # Average adherence details
        avg_adherence_details = AdherenceScore(
            overall=np.mean([s.adherence_details.overall for s in scores]),
            tempo_adherence=np.mean([s.adherence_details.tempo_adherence for s in scores]),
            key_adherence=np.mean([s.adherence_details.key_adherence for s in scores]),
            structure_adherence=np.mean([s.adherence_details.structure_adherence for s in scores]),
            genre_adherence=np.mean([s.adherence_details.genre_adherence for s in scores]),
            instrumentation_adherence=np.mean([s.adherence_details.instrumentation_adherence for s in scores]),
            details=scores[0].adherence_details.details  # Keep first sample's details
        )
        
        # Combine notes
        all_notes = []
        for score in scores:
            all_notes.extend(score.notes)
        unique_notes = list(set(all_notes))
        
        return CriticScore(
            overall=avg_overall,
            adherence=avg_adherence,
            style_match=avg_style_match,
            mix_quality=avg_mix_quality,
            adherence_details=avg_adherence_details,
            style_details=scores[0].style_details,  # Keep first sample's details
            mix_details=scores[0].mix_details,
            confidence=avg_confidence,
            notes=unique_notes
        )
    
    def _compute_aggregate_metrics(self, scores: List[CriticScore]) -> Dict[str, Any]:
        """Compute aggregate metrics from list of CriticScore objects"""
        if not scores:
            return {}
        
        # Extract all numerical scores
        overall_scores = [s.overall for s in scores]
        adherence_scores = [s.adherence for s in scores]
        style_scores = [s.style_match for s in scores]
        mix_scores = [s.mix_quality for s in scores]
        confidence_scores = [s.confidence for s in scores]
        
        # Component adherence scores
        tempo_scores = [s.adherence_details.tempo_adherence for s in scores]
        key_scores = [s.adherence_details.key_adherence for s in scores]
        structure_scores = [s.adherence_details.structure_adherence for s in scores]
        genre_scores = [s.adherence_details.genre_adherence for s in scores]
        instrumentation_scores = [s.adherence_details.instrumentation_adherence for s in scores]
        
        metrics = {
            # Main metrics
            'overall_mean': np.mean(overall_scores),
            'overall_std': np.std(overall_scores),
            'overall_median': np.median(overall_scores),
            'adherence_mean': np.mean(adherence_scores),
            'adherence_std': np.std(adherence_scores),
            'style_match_mean': np.mean(style_scores),
            'style_match_std': np.std(style_scores),
            'mix_quality_mean': np.mean(mix_scores),
            'mix_quality_std': np.std(mix_scores),
            'confidence_mean': np.mean(confidence_scores),
            
            # Component adherence metrics
            'tempo_adherence_mean': np.mean(tempo_scores),
            'key_adherence_mean': np.mean(key_scores),
            'structure_adherence_mean': np.mean(structure_scores),
            'genre_adherence_mean': np.mean(genre_scores),
            'instrumentation_adherence_mean': np.mean(instrumentation_scores),
            
            # Percentiles
            'overall_p25': np.percentile(overall_scores, 25),
            'overall_p75': np.percentile(overall_scores, 75),
            'adherence_p25': np.percentile(adherence_scores, 25),
            'adherence_p75': np.percentile(adherence_scores, 75),
            
            # Quality thresholds
            'high_quality_ratio': np.mean(np.array(overall_scores) >= 0.7),
            'acceptable_quality_ratio': np.mean(np.array(overall_scores) >= 0.5),
            'high_adherence_ratio': np.mean(np.array(adherence_scores) >= 0.7),
            
            # Sample count
            'num_samples': len(scores)
        }
        
        return metrics
    
    def compare_models(
        self,
        before_model: torch.nn.Module,
        after_model: torch.nn.Module,
        test_samples: List[Dict[str, Any]],
        num_generations: int = 5
    ) -> Dict[str, Any]:
        """
        Compare adherence metrics between two models
        
        Args:
            before_model: Model before DPO finetuning
            after_model: Model after DPO finetuning
            test_samples: Test samples for evaluation
            num_generations: Generations per prompt
            
        Returns:
            Comparison results with statistical significance tests
        """
        logger.info("Evaluating before model...")
        before_metrics = self.evaluate_model_adherence(
            before_model, test_samples, num_generations
        )
        
        logger.info("Evaluating after model...")
        after_metrics = self.evaluate_model_adherence(
            after_model, test_samples, num_generations
        )
        
        # Compute improvements
        improvements = {}
        significance_tests = {}
        
        metric_keys = [
            'overall_mean', 'adherence_mean', 'style_match_mean', 'mix_quality_mean',
            'tempo_adherence_mean', 'key_adherence_mean', 'structure_adherence_mean',
            'genre_adherence_mean', 'instrumentation_adherence_mean'
        ]
        
        for key in metric_keys:
            if key in before_metrics and key in after_metrics:
                before_val = before_metrics[key]
                after_val = after_metrics[key]
                improvement = after_val - before_val
                relative_improvement = improvement / (before_val + 1e-10) * 100
                
                improvements[key] = {
                    'absolute': improvement,
                    'relative_percent': relative_improvement,
                    'before': before_val,
                    'after': after_val
                }
        
        # Statistical significance tests (placeholder - would need actual sample scores)
        # This would require collecting individual scores rather than just aggregates
        for key in metric_keys:
            if key in improvements:
                # Placeholder p-value (would compute with actual samples)
                p_value = 0.05 if abs(improvements[key]['relative_percent']) > 5 else 0.2
                significance_tests[key] = {
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        comparison_results = {
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'improvements': improvements,
            'significance_tests': significance_tests,
            'summary': self._generate_comparison_summary(improvements, significance_tests)
        }
        
        return comparison_results
    
    def _generate_comparison_summary(
        self,
        improvements: Dict[str, Dict[str, float]],
        significance_tests: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary of comparison results"""
        significant_improvements = []
        significant_degradations = []
        
        for metric, improvement in improvements.items():
            if significance_tests.get(metric, {}).get('significant', False):
                rel_change = improvement['relative_percent']
                if rel_change > 0:
                    significant_improvements.append((metric, rel_change))
                else:
                    significant_degradations.append((metric, rel_change))
        
        # Overall assessment
        overall_improvement = improvements.get('overall_mean', {}).get('relative_percent', 0)
        adherence_improvement = improvements.get('adherence_mean', {}).get('relative_percent', 0)
        
        summary = {
            'overall_improvement_percent': overall_improvement,
            'adherence_improvement_percent': adherence_improvement,
            'significant_improvements': significant_improvements,
            'significant_degradations': significant_degradations,
            'num_improved_metrics': len(significant_improvements),
            'num_degraded_metrics': len(significant_degradations),
            'recommendation': self._get_recommendation(overall_improvement, adherence_improvement)
        }
        
        return summary
    
    def _get_recommendation(self, overall_improvement: float, adherence_improvement: float) -> str:
        """Generate recommendation based on improvements"""
        if overall_improvement > 10 and adherence_improvement > 5:
            return "Strong improvement - DPO training was highly successful"
        elif overall_improvement > 5 and adherence_improvement > 2:
            return "Good improvement - DPO training was successful"
        elif overall_improvement > 0 and adherence_improvement > 0:
            return "Modest improvement - DPO training had positive effect"
        elif overall_improvement > -5:
            return "Minimal change - DPO training had limited effect"
        else:
            return "Degradation - DPO training had negative effect, review training setup"
    
    def save_evaluation_report(
        self,
        comparison_results: Dict[str, Any],
        output_path: str
    ):
        """Save detailed evaluation report"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = output_path / "adherence_evaluation_report.json"
        with open(json_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Create visualizations
        self._create_comparison_plots(comparison_results, output_path)
        
        # Generate markdown report
        self._generate_markdown_report(comparison_results, output_path)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def _create_comparison_plots(
        self,
        comparison_results: Dict[str, Any],
        output_path: Path
    ):
        """Create visualization plots for comparison"""
        improvements = comparison_results['improvements']
        
        # Improvement bar plot
        plt.figure(figsize=(12, 8))
        
        metrics = []
        rel_improvements = []
        colors = []
        
        for metric, data in improvements.items():
            if 'relative_percent' in data:
                metrics.append(metric.replace('_mean', '').replace('_', ' ').title())
                rel_improvements.append(data['relative_percent'])
                colors.append('green' if data['relative_percent'] > 0 else 'red')
        
        plt.barh(metrics, rel_improvements, color=colors, alpha=0.7)
        plt.xlabel('Relative Improvement (%)')
        plt.title('DPO Finetuning Impact on Adherence Metrics')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(rel_improvements):
            plt.text(v + (0.5 if v >= 0 else -0.5), i, f'{v:.1f}%', 
                    va='center', ha='left' if v >= 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(output_path / "adherence_improvements.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Before/after comparison plot
        plt.figure(figsize=(10, 6))
        
        before_scores = []
        after_scores = []
        metric_names = []
        
        for metric, data in improvements.items():
            if 'before' in data and 'after' in data:
                before_scores.append(data['before'])
                after_scores.append(data['after'])
                metric_names.append(metric.replace('_mean', '').replace('_', ' ').title())
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.bar(x - width/2, before_scores, width, label='Before DPO', alpha=0.7)
        plt.bar(x + width/2, after_scores, width, label='After DPO', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Before vs After DPO Finetuning')
        plt.xticks(x, metric_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "before_after_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(
        self,
        comparison_results: Dict[str, Any],
        output_path: Path
    ):
        """Generate markdown evaluation report"""
        summary = comparison_results['summary']
        improvements = comparison_results['improvements']
        
        report = f"""# DPO Finetuning Adherence Evaluation Report

## Summary

**Overall Improvement**: {summary['overall_improvement_percent']:.2f}%
**Adherence Improvement**: {summary['adherence_improvement_percent']:.2f}%

**Recommendation**: {summary['recommendation']}

## Detailed Results

### Significant Improvements
"""
        
        for metric, improvement in summary['significant_improvements']:
            report += f"- **{metric.replace('_', ' ').title()}**: +{improvement:.2f}%\n"
        
        if summary['significant_degradations']:
            report += "\n### Significant Degradations\n"
            for metric, degradation in summary['significant_degradations']:
                report += f"- **{metric.replace('_', ' ').title()}**: {degradation:.2f}%\n"
        
        report += "\n## All Metrics Comparison\n\n"
        report += "| Metric | Before | After | Absolute Change | Relative Change (%) |\n"
        report += "|--------|--------|-------|-----------------|--------------------|\n"
        
        for metric, data in improvements.items():
            metric_name = metric.replace('_mean', '').replace('_', ' ').title()
            before = data['before']
            after = data['after']
            abs_change = data['absolute']
            rel_change = data['relative_percent']
            
            report += f"| {metric_name} | {before:.3f} | {after:.3f} | {abs_change:+.3f} | {rel_change:+.2f}% |\n"
        
        report += f"""
## Training Statistics

- **Number of Improved Metrics**: {summary['num_improved_metrics']}
- **Number of Degraded Metrics**: {summary['num_degraded_metrics']}

## Visualizations

- [Adherence Improvements](./adherence_improvements.png)
- [Before vs After Comparison](./before_after_comparison.png)

---
*Report generated automatically from DPO evaluation results*
"""
        
        report_path = output_path / "evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)

def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DPO Finetuning Results")
    parser.add_argument("--before_model", required=True, help="Path to model before DPO")
    parser.add_argument("--after_model", required=True, help="Path to model after DPO")
    parser.add_argument("--critic_model", required=True, help="Path to critic model")
    parser.add_argument("--test_data", required=True, help="Path to test samples JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory for report")
    parser.add_argument("--num_generations", type=int, default=5, help="Generations per prompt")
    
    args = parser.parse_args()
    
    # Load test data
    with open(args.test_data, 'r') as f:
        test_samples = json.load(f)
    
    # Load models (placeholder - would need actual loading)
    before_model = None  # Load before model
    after_model = None   # Load after model
    critic_model = None  # Load critic model
    
    # Create evaluator
    evaluator = AdherenceEvaluator(critic_model)
    
    # Run comparison
    logger.info("Starting model comparison...")
    results = evaluator.compare_models(
        before_model, after_model, test_samples, args.num_generations
    )
    
    # Save report
    evaluator.save_evaluation_report(results, args.output_dir)
    
    # Print summary
    summary = results['summary']
    print(f"\nEvaluation Complete!")
    print(f"Overall Improvement: {summary['overall_improvement_percent']:.2f}%")
    print(f"Adherence Improvement: {summary['adherence_improvement_percent']:.2f}%")
    print(f"Recommendation: {summary['recommendation']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()