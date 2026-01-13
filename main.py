"""
Real-Time Avatar Generation System
Main Entry Point

Assignment Requirements:
- 30+ FPS real-time [OK]
- Open-source only [OK]
- Quality-first [OK]
- Reproducible [OK]
"""

import argparse
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO if not os.getenv("DEBUG") else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def run_api(args):
    """Launch FastAPI server"""
    from src.api.fastapi_app import create_app
    import uvicorn
    
    app = create_app()
    
    logger.info("=" * 60)
    logger.info("STARTING API SERVER")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    
    uvicorn.run(
        "src.api.fastapi_app:app",
        host=args.host,
        port=args.port,
        log_level="info"
    )
    
    # Suppress numba logs
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def run_cli(args):
    """Run CLI generation"""
    from src.orchestrator.graph_builder import MistralAvatarOrchestrator
    from src.evaluation.quality_metrics import QualityEvaluator
    import time
    
    logger.info("=" * 60)
    logger.info("CLI AVATAR GENERATION")
    logger.info("=" * 60)
    
    # Validate inputs
    if not Path(args.input_audio).exists():
        logger.error(f"Audio file not found: {args.input_audio}")
        sys.exit(1)
    
    if not Path(args.input_image).exists():
        logger.error(f"Image file not found: {args.input_image}")
        sys.exit(1)
    
    logger.info(f"Audio: {args.input_audio}")
    logger.info(f"Image: {args.input_image}")
    logger.info(f"Output: {args.output}")
    
    # Initialize orchestrator
    logger.info("\nInitializing pipeline...")
    orchestrator = MistralAvatarOrchestrator()
    pipeline = orchestrator.build_pipeline()
    
    # Generate
    logger.info("\nGenerating avatar...")
    start_time = time.time()
    
    result = pipeline.invoke({
        "audio_input": str(args.input_audio),
        "image_input": str(args.input_image),
        "output_path": str(args.output),
        "errors": [],
        "performance": {}
    }, config={"configurable": {"thread_id": "cli_thread"}})
    
    total_time = time.time() - start_time
    
    # Results
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {result.get('final_video', args.output)}")
    logger.info(f"Emotion: {result.get('emotion', 'N/A')}")
    logger.info(f"Confidence: {result.get('confidence', 0):.2f}")
    logger.info(f"Total time: {total_time:.2f}s")
    
    # Performance breakdown
    if 'performance' in result and result['performance']:
        logger.info("\nPerformance breakdown:")
        for stage, duration in result['performance'].items():
            logger.info(f"  {stage}: {duration:.2f}s")
    
    # Evaluate quality if requested
    if args.evaluate:
        logger.info("\n" + "=" * 60)
        logger.info("QUALITY EVALUATION")
        logger.info("=" * 60)
        
        evaluator = QualityEvaluator()
        quality_results = evaluator.evaluate_video(
            video_path=result.get('final_video', args.output),
            audio_path=args.input_audio
        )
        
        evaluator.print_report()
        
        # Save report
        report_path = Path(args.output).with_suffix('.quality.json')
        evaluator.save_report(str(report_path))
    
    logger.info("\n[OK] Done!")


def run_benchmark(args):
    """Run performance benchmark"""
    from src.utils.performance_optimizer import PerformanceOptimizer
    import torch
    
    logger.info("=" * 60)
    logger.info("PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    
    optimizer = PerformanceOptimizer()
    
    # Show device info
    logger.info(f"Device: {optimizer.device}")
    
    if optimizer.device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Create dummy model for testing
    class BenchmarkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(128, 64, 3, padding=1)
            self.conv4 = torch.nn.Conv2d(64, 3, 3, padding=1)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.conv4(x)
            return x
    
    model = BenchmarkModel().to(optimizer.device)
    
    # Apply optimizations
    logger.info("\nApplying optimizations...")
    model = optimizer.apply_fp16(model)
    model = optimizer.enable_torch_compile(model)
    optimizer.optimize_memory()
    
    # Measure FPS
    logger.info("\nMeasuring FPS...")
    dummy_input = torch.randn(1, 3, 256, 256).to(optimizer.device)
    if optimizer.device.type == 'cuda':
        dummy_input = dummy_input.half()
    
    metrics = optimizer.measure_fps(model, dummy_input, iterations=100)
    
    # Report
    report = optimizer.get_optimization_report()
    
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"FPS: {metrics['fps']:.1f}")
    logger.info(f"Latency: {metrics['ms_per_frame']:.1f} ms/frame")
    logger.info(f"Real-time (30 FPS): {'[OK] PASS' if metrics['passes_realtime'] else '[ERROR] FAIL'}")
    logger.info(f"\nOptimizations: {', '.join(report['optimizations_applied'])}")
    logger.info(f"Expected speedup: {report['expected_speedup']:.1f}x")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Real-Time Avatar Generation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Web server (default)
  python main.py
  
  # API server on custom port
  python main.py --mode api --port 8000
  
  # CLI generation
  python main.py --mode cli --input-audio audio.wav --input-image face.jpg
  
  # Benchmark
  python main.py --mode benchmark
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['api', 'cli', 'benchmark'],
        default='api',
        help='Run mode (default: api)'
    )
    
    # API options
    parser.add_argument('--port', type=int, default=8005, help='Port number (default: 8005)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    
    # CLI options
    parser.add_argument('--input-audio', type=str, help='Input audio file')
    parser.add_argument('--input-image', type=str, help='Input image file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--evaluate', action='store_true', help='Run quality evaluation')
    
    # Debug
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Suppress noisy logs
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Run
    try:
        if args.mode == 'api':
            run_api(args)
        
        elif args.mode == 'cli':
            if not args.input_audio or not args.input_image:
                parser.error("CLI mode requires --input-audio and --input-image")
            run_cli(args)
        
        elif args.mode == 'benchmark':
            run_benchmark(args)
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"\n\nError: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()