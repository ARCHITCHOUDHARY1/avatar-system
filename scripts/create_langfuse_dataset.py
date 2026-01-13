"""
Langfuse Dataset Management for Avatar System
Creates and manages evaluation datasets for avatar generation quality tracking
"""

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

class AvatarDatasetManager:
    """Manage Langfuse datasets for avatar evaluation"""
    
    def __init__(self):
        """Initialize Langfuse client"""
        self.langfuse = Langfuse()
        self.dataset_name = "avatar-eval-dataset"
    
    def create_dataset(self):
        """Create the main evaluation dataset"""
        try:
            self.langfuse.create_dataset(
                name=self.dataset_name,
                description="Avatar generation evaluation dataset with test cases for quality tracking",
                metadata={
                    "project": "avatar-system-orchestrator",
                    "version": "1.0",
                    "created_by": "dataset_manager"
                }
            )
            print(f"? Created dataset: {self.dataset_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"??  Dataset already exists: {self.dataset_name}")
            else:
                print(f"? Error creating dataset: {e}")
                raise
    
    def add_test_case(
        self,
        audio_path: str,
        image_path: str,
        expected_transcript: str = None,
        expected_emotion: str = None,
        min_quality_score: float = 0.8,
        test_name: str = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Add a test case to the dataset
        
        Args:
            audio_path: Path to audio file
            image_path: Path to image/video file
            expected_transcript: Expected transcription
            expected_emotion: Expected emotion detection
            min_quality_score: Minimum acceptable quality score
            test_name: Name for this test case
            metadata: Additional metadata
        """
        input_data = {
            "audio_path": audio_path,
            "image_path": image_path,
            "prompt": f"Generate avatar from audio and image"
        }
        
        expected_output = {
            "expected_transcript": expected_transcript,
            "expected_emotion": expected_emotion,
            "min_quality_score": min_quality_score,
            "min_lip_sync_score": 0.85,
            "min_face_similarity": 0.90
        }
        
        item_metadata = {
            "test_name": test_name or f"test_{Path(audio_path).stem}",
            "audio_file": Path(audio_path).name,
            "image_file": Path(image_path).name,
            **(metadata or {})
        }
        
        try:
            self.langfuse.create_dataset_item(
                dataset_name=self.dataset_name,
                input=json.dumps(input_data, indent=2),
                expected_output=json.dumps(expected_output, indent=2),
                metadata=item_metadata
            )
            print(f"? Added test case: {item_metadata['test_name']}")
        except Exception as e:
            print(f"? Error adding test case: {e}")
            raise
    
    def add_sample_test_cases(self):
        """Add sample test cases for demonstration"""
        
        test_cases = [
            {
                "audio_path": "data/inputs/sample_happy.wav",
                "image_path": "data/inputs/face_neutral.jpg",
                "expected_transcript": "Hello, I'm happy to see you!",
                "expected_emotion": "happy",
                "min_quality_score": 0.85,
                "test_name": "happy_emotion_test",
                "metadata": {"category": "emotion", "difficulty": "easy"}
            },
            {
                "audio_path": "data/inputs/sample_sad.wav",
                "image_path": "data/inputs/face_neutral.jpg",
                "expected_transcript": "I'm feeling a bit down today.",
                "expected_emotion": "sad",
                "min_quality_score": 0.80,
                "test_name": "sad_emotion_test",
                "metadata": {"category": "emotion", "difficulty": "medium"}
            },
            {
                "audio_path": "data/inputs/sample_neutral.wav",
                "image_path": "data/inputs/face_smiling.jpg",
                "expected_transcript": "This is a neutral statement.",
                "expected_emotion": "neutral",
                "min_quality_score": 0.90,
                "test_name": "neutral_baseline_test",
                "metadata": {"category": "baseline", "difficulty": "easy"}
            },
            {
                "audio_path": "data/inputs/long_speech.wav",
                "image_path": "data/inputs/face_professional.jpg",
                "expected_transcript": "This is a longer speech for testing performance...",
                "expected_emotion": "neutral",
                "min_quality_score": 0.75,
                "test_name": "long_speech_performance_test",
                "metadata": {"category": "performance", "difficulty": "hard"}
            },
            {
                "audio_path": "data/inputs/noisy_audio.wav",
                "image_path": "data/inputs/face_neutral.jpg",
                "expected_transcript": "Testing with background noise.",
                "expected_emotion": "neutral",
                "min_quality_score": 0.70,
                "test_name": "noisy_audio_robustness_test",
                "metadata": {"category": "robustness", "difficulty": "hard"}
            }
        ]
        
        print(f"\nAdding {len(test_cases)} sample test cases...")
        for test_case in test_cases:
            self.add_test_case(**test_case)
        
        print(f"\n? Added {len(test_cases)} test cases to dataset")
    
    def list_dataset_items(self):
        """List all items in the dataset"""
        try:
            dataset = self.langfuse.get_dataset(self.dataset_name)
            items = dataset.items
            
            print(f"\n? Dataset: {self.dataset_name}")
            print(f"Total items: {len(items)}")
            print("\nTest Cases:")
            
            for i, item in enumerate(items, 1):
                metadata = item.metadata or {}
                print(f"\n{i}. {metadata.get('test_name', 'Unnamed')}")
                print(f"   Category: {metadata.get('category', 'N/A')}")
                print(f"   Difficulty: {metadata.get('difficulty', 'N/A')}")
                
                # Parse input
                try:
                    input_data = json.loads(item.input)
                    print(f"   Audio: {input_data.get('audio_path', 'N/A')}")
                    print(f"   Image: {input_data.get('image_path', 'N/A')}")
                except:
                    pass
            
            return items
            
        except Exception as e:
            print(f"? Error listing dataset items: {e}")
            return []
    
    def get_dataset_statistics(self):
        """Get statistics about the dataset"""
        try:
            items = self.list_dataset_items()
            
            if not items:
                return
            
            categories = {}
            difficulties = {}
            
            for item in items:
                metadata = item.metadata or {}
                cat = metadata.get('category', 'unknown')
                diff = metadata.get('difficulty', 'unknown')
                
                categories[cat] = categories.get(cat, 0) + 1
                difficulties[diff] = difficulties.get(diff, 0) + 1
            
            print("\n? Dataset Statistics:")
            print(f"\nBy Category:")
            for cat, count in categories.items():
                print(f"  - {cat}: {count}")
            
            print(f"\nBy Difficulty:")
            for diff, count in difficulties.items():
                print(f"  - {diff}: {count}")
            
        except Exception as e:
            print(f"? Error getting statistics: {e}")


def main():
    """Main function to demonstrate dataset management"""
    
    print("=" * 60)
    print("LANGFUSE DATASET MANAGEMENT - AVATAR SYSTEM")
    print("=" * 60)
    
    # Initialize manager
    manager = AvatarDatasetManager()
    
    # Create dataset
    print("\n1. Creating dataset...")
    manager.create_dataset()
    
    # Add sample test cases
    print("\n2. Adding sample test cases...")
    manager.add_sample_test_cases()
    
    # List items
    print("\n3. Listing dataset items...")
    manager.list_dataset_items()
    
    # Get statistics
    print("\n4. Dataset statistics...")
    manager.get_dataset_statistics()
    
    print("\n" + "=" * 60)
    print("? DATASET SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. View dataset at: https://cloud.langfuse.com")
    print("2. Navigate to: Datasets ? avatar-eval-dataset")
    print("3. Run evaluations with: python scripts/run_dataset_evaluation.py")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
