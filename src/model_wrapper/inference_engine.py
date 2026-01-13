
import torch
import numpy as np
from typing import Dict, Optional, List
import logging
import time

logger = logging.getLogger(__name__)


class InferenceEngine:
    
    def __init__(
        self,
        device: str = "cuda",
        use_fp16: bool = True,
        use_amp: bool = False,
        compile_model: bool = False
    ):
        self.device = device
        self.use_fp16 = use_fp16
        self.use_amp = use_amp
        self.compile_model = compile_model
        
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        
        # Move to device
        model = model.to(self.device)
        
        # Half precision
        if self.use_fp16 and self.device == "cuda":
            model = model.half()
            logger.info("Using FP16 precision")
        
        # Compile model (PyTorch 2.0+)
        if self.compile_model:
            try:
                model = torch.compile(model)
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Set to eval mode
        model.eval()
        
        return model
    
    @torch.no_grad()
    def infer(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        batch_size: int = 1
    ) -> torch.Tensor:
        
        outputs = []
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch_inputs = {
                k: v[i:i+batch_size].to(self.device)
                for k, v in inputs.items()
            }
            
            # Run inference with AMP if enabled
            if self.use_amp and self.device == "cuda":
                with torch.cuda.amp.autocast():
                    batch_output = model(**batch_inputs)
            else:
                batch_output = model(**batch_inputs)
            
            outputs.append(batch_output.cpu())
        
        return torch.cat(outputs, dim=0)
    
    def warmup(self, model: torch.nn.Module, input_shape: tuple):
        logger.info("Warming up model...")
        
        dummy_input = torch.randn(input_shape).to(self.device)
        
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        logger.info("Model warmup complete")
    
    def benchmark(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        self.warmup(model, input_shape)
        
        # Benchmark
        times = []
        
        for _ in range(num_iterations):
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "fps": 1.0 / np.mean(times)
        }
