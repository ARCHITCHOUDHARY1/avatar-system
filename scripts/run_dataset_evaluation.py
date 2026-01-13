"""
Run Avatar System Evaluation on Langfuse Dataset
Evaluates avatar generation quality against test cases in Langfuse dataset
"""

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
import json
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.pipeline_runner import PipelineRunner
from src.observability import is_langfuse_enabled


class AvatarEvaluator:
    """Evaluate avatar generation against Langfuse dataset"""
    
    def __init__(self, dataset_name: str = "avatar-eval-dataset"):
        """Initialize evaluator"""
        self.langfuse = Langfuse()
        self.dataset_name = dataset_name
        self.runner = PipelineRunner()
    
    def run_evaluation(self, limit: int = None):
        """
        Run evaluation on dataset items
        
        Args:
            limit: Maximum number of items to evaluate (None = all)
        """
        print("=" * 60)
        print("AVATAR SYSTEM EVALUATION")
        print("=" * 60)
        
        # Get dataset
        try:
            dataset = self.langfuse.get_dataset(self.dataset_name)
            items = dataset.items
            
            if limit:
                items = items[:limit]
            
            print(f"\nDataset: {self.dataset_name}")
            print(f"Total items: {len(items)}")
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"? Error loading dataset: {e}")
            return
        
        # Run evaluation on each item
        results = []
        
        for i, item in enumerate(items, 1):
            print(f"\n[{i}/{len(items)}] Evaluating test case...")
            
            try:
                result = self.evaluate_item(item)
                results.append(result)
                
                # Print result
                status = "? PASSED" if result['passed'] else "? FAILED"
                print(f"   {status}")
                print(f"   Quality Score: {result.get('quality_score', 'N/A')}")
                
            except Exception as e:
                print(f"   ? Error: {e}")
                results.append({
                    'passed': False,
                    'error': str(e)
                })
        
        # Print summary
        self.print_summary(results)
    
    def evaluate_item(self, item):
        """
        Evaluate a single dataset item
        
        Args:
            item: Dataset item from Langfuse
        
        Returns:
            dict: Evaluation result
        """
        # Parse input and expected output
        input_data = json.loads(item.input)
        expected_output = json.loads(item.expected_output)
        
        metadata = item.metadata or {}
        test_name = metadata.get('test_name', 'unknown')
        
        print(f"   Test: {test_name}")
        print(f"   Audio: {input_data['audio_path']}")
        print(f"   Image: {input_data['image_path']}")
        
        # Create run in Langfuse
        run = item.link(
            trace_name=f"eval_{test_name}",
            run_name=f"run_{int(time.time())}",
            run_metadata={
                "test_name": test_name,
                "category": metadata.get('category'),
                "difficulty": metadata.get('difficulty')
            }
        )
        
        # Run pipeline
        start_time = time.time()
        
        try:
            result = self.runner.process(
                audio_path=input_data['audio_path'],
                image_path=input_data['image_path'],
                output_path=f"data/outputs/eval_{test_name}.mp4",
                session_id=f"eval_{test_name}_{int(time.time())}"
            )
            
            duration = time.time() - start_time
            
            # Evaluate result
            evaluation = self.evaluate_result(result, expected_output)
            evaluation['duration'] = duration
            evaluation['test_name'] = test_name
            
            # Log scores to Langfuse
            if evaluation['passed']:
                run.score(
                    name="quality_score",
                    value=evaluation['quality_score']
                )
                run.score(
                    name="overall_pass",
                    value=1.0
                )
            else:
                run.score(
                    name="overall_pass",
                    value=0.0
                )
            
            return evaluation
            
        except Exception as e:
            duration = time.time() - start_time
            
            run.score(
                name="overall_pass",
                value=0.0
            )
            
            return {
                'passed': False,
                'error': str(e),
                'duration': duration,
                'test_name': test_name
            }
    
    def evaluate_result(self, result, expected_output):
        """
        Evaluate pipeline result against expected output
        
        Args:
            result: Pipeline result
            expected_output: Expected output from dataset
        
        Returns:
            dict: Evaluation metrics
        """
        evaluation = {
            'passed': True,
            'quality_score': 0.0,
            'checks': {}
        }
        
        # Check for errors
        if result.get('errors'):
            evaluation['passed'] = False
            evaluation['checks']['no_errors'] = False
            return evaluation
        else:
            evaluation['checks']['no_errors'] = True
        
        # Check transcript (if available)
        if 'expected_transcript' in expected_output and expected_output['expected_transcript']:
            actual_transcript = result.get('transcribed_text', '')
            expected_transcript = expected_output['expected_transcript']
            
            # Simple similarity check (you can use more sophisticated methods)
            transcript_match = expected_transcript.lower() in actual_transcript.lower()
            evaluation['checks']['transcript_match'] = transcript_match
            
            if not transcript_match:
                evaluation['passed'] = False
        
        # Check emotion (if available)
        if 'expected_emotion' in expected_output and expected_output['expected_emotion']:
            actual_emotion = result.get('emotion', 'unknown')
            expected_emotion = expected_output['expected_emotion']
            
            emotion_match = actual_emotion.lower() == expected_emotion.lower()
            evaluation['checks']['emotion_match'] = emotion_match
            
            if not emotion_match:
                evaluation['passed'] = False
        
        # Calculate quality score
        checks_passed = sum(1 for v in evaluation['checks'].values() if v)
        total_checks = len(evaluation['checks'])
        
        if total_checks > 0:
            evaluation['quality_score'] = checks_passed / total_checks
        else:
            evaluation['quality_score'] = 1.0  # No checks = pass
        
        # Check minimum quality threshold
        min_quality = expected_output.get('min_quality_score', 0.8)
        if evaluation['quality_score'] < min_quality:
            evaluation['passed'] = False
        
        return evaluation
    
    def print_summary(self, results):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        total = len(results)
        passed = sum(1 for r in results if r.get('passed', False))
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"? Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"? Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Average quality score
        quality_scores = [r.get('quality_score', 0) for r in results if 'quality_score' in r]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            print(f"\n? Average Quality Score: {avg_quality:.2f}")
        
        # Average duration
        durations = [r.get('duration', 0) for r in results if 'duration' in r]
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"??  Average Duration: {avg_duration:.2f}s")
        
        # Failed tests
        if failed > 0:
            print("\n? Failed Tests:")
            for r in results:
                if not r.get('passed', False):
                    print(f"   - {r.get('test_name', 'unknown')}")
                    if 'error' in r:
                        print(f"     Error: {r['error']}")
        
        print("\n" + "=" * 60)
        print("? VIEW DETAILED RESULTS IN LANGFUSE")
        print("=" * 60)
        print("\n1. Go to: https://cloud.langfuse.com")
        print("2. Navigate to: Datasets ? avatar-eval-dataset")
        print("3. Click on 'Runs' to see all evaluation runs")
        print("4. View scores, traces, and detailed metrics")
        print("\n" + "=" * 60)


def main():
    """Main function"""
    
    if not is_langfuse_enabled():
        print("? Langfuse is not enabled")
        print("   Set ENABLE_LANGFUSE=true in .env")
        return
    
    evaluator = AvatarEvaluator()
    
    # Run evaluation on first 3 items (for demo)
    # Remove limit to evaluate all items
    evaluator.run_evaluation(limit=3)


if __name__ == "__main__":
    main()
