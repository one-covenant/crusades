#!/usr/bin/env python3
"""
Download benchmark model and data for Templar Tournament.

CRITICAL: This downloads the EXACT 8B model and dataset ALL participants use.
Everyone competes on identical resources for fairness.

Model: meta-llama/Llama-3.2-8B (specified in hparams.json)
Data: HuggingFaceFW/fineweb (specified in hparams.json)
"""

import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config():
    """Load benchmark configuration from hparams.json."""
    config_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    with open(config_path) as f:
        return json.load(f)


def download_model(model_dir: Path, config: dict) -> None:
    """Download the official 8B model.
    
    Args:
        model_dir: Directory to save model
        config: Configuration from hparams.json
    """
    model_name = config["benchmark_model_name"]
    model_revision = config.get("benchmark_model_revision", "main")
    
    logger.info(f"📥 Downloading official benchmark model:")
    logger.info(f"   {model_name}")
    logger.info(f"   Revision: {model_revision}")
    logger.info(f"   Size: ~7B parameters (~15GB)")
    logger.info("")
    logger.info("⏳ This will take 10-30 minutes...")
    logger.info("")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,  # Required for some models like Qwen
        )
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            trust_remote_code=True,
        )
        
        # Save to local directory
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model downloaded: {num_params:,} parameters")
        logger.info(f"📁 Saved to: {model_dir}")
        
    except ImportError:
        logger.error("❌ transformers library not found")
        logger.error("   Run: uv sync")
        raise
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.error("")
        logger.error("💡 Troubleshooting:")
        logger.error("   1. Some models require HuggingFace access")
        logger.error("   2. Run: huggingface-cli login")
        logger.error("   3. Request access to model if needed")
        raise


def download_data(data_path: Path, config: dict) -> None:
    """Download and prepare the official dataset.
    
    Args:
        data_path: Path to save prepared data
        config: Configuration from hparams.json
    """
    dataset_name = config["benchmark_dataset_name"]
    dataset_split = config["benchmark_dataset_split"]
    dataset_size = config.get("benchmark_dataset_size", 100000)
    seq_length = config["benchmark_sequence_length"]
    model_name = config["benchmark_model_name"]
    
    logger.info(f"📥 Preparing official benchmark data:")
    logger.info(f"   Dataset: {dataset_name}")
    logger.info(f"   Split: {dataset_split}[:{dataset_size}]")
    logger.info(f"   Samples: {dataset_size:,}")
    logger.info(f"   Sequence length: {seq_length}")
    logger.info("")
    logger.info("⏳ This will take 10-20 minutes...")
    logger.info("")
    
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset with streaming to avoid downloading entire dataset
        # With 5 steps * 8 batch = 40 samples per eval
        # 100k samples = ~2,500 unique evaluations before repeating
        logger.info("⏳ Loading dataset (streaming first 100k samples)...")
        dataset = load_dataset(
            dataset_name,
            split=dataset_split,
            streaming=True,  # Stream to avoid downloading entire 59TB dataset!
        )
        
        logger.info("⏳ Tokenizing sequences...")
        tokenized_samples = []
        
        for i, item in enumerate(dataset):
            if i >= dataset_size:  # Stop at 100k
                break
            
            # Get text field
            text = item.get("text", item.get("content", ""))
            if not text:
                continue
            
            # Tokenize
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_length,
                truncation=True,
                padding="max_length",
            )["input_ids"][0]
            
            tokenized_samples.append(tokens)
            
            if (i + 1) % 10000 == 0:
                logger.info(f"   Tokenized {i+1}/{dataset_size} samples...")
        
        # Stack into tensor
        data_tensor = torch.stack(tokenized_samples)
        
        # Save
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_tensor, data_path)
        
        logger.info(f"✅ Data prepared: {len(tokenized_samples)} samples")
        logger.info(f"   Total tokens: {data_tensor.numel():,}")
        logger.info(f"📁 Saved to: {data_path}")
        
    except ImportError:
        logger.error("❌ datasets library not found")
        logger.error("   Run: uv sync")
        raise
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        raise


def main():
    logger.info("="*70)
    logger.info("TEMPLAR TOURNAMENT - BENCHMARK SETUP")
    logger.info("="*70)
    logger.info("")
    logger.info("This downloads the OFFICIAL 7B model and dataset that:")
    logger.info("  ✓ ALL miners use for local testing")
    logger.info("  ✓ ALL validators use for evaluation")
    logger.info("")
    logger.info("Requirements:")
    logger.info("  • ~20GB free disk space")
    logger.info("  • Stable internet connection")
    logger.info("")
    
    # Load configuration
    config = load_config()
    
    model_dir = Path("benchmark/model")
    data_path = Path("benchmark/data/train.pt")
    
    logger.info("📋 Configuration (from hparams.json):")
    logger.info(f"   Model: {config['benchmark_model_name']}")
    logger.info(f"   Dataset: {config['benchmark_dataset_name']}")
    logger.info(f"   Batch size: {config['benchmark_batch_size']}")
    logger.info(f"   Sequence length: {config['benchmark_sequence_length']}")
    logger.info(f"   Training steps per eval: {config['eval_steps']}")
    logger.info("")
    
    input("Press ENTER to start download...")
    logger.info("")
    
    # Download model
    download_model(model_dir, config)
    logger.info("")
    
    # Download and prepare data
    download_data(data_path, config)
    logger.info("")
    
    logger.info("="*70)
    logger.info("✅ SETUP COMPLETE")
    logger.info("="*70)
    logger.info("")
    logger.info("📁 Files created:")
    logger.info(f"   Model: {model_dir}/")
    logger.info(f"   Data:  {data_path}")
    logger.info("")
    logger.info("✅ You now have the SAME resources validators use")
    logger.info("")
    logger.info("📝 Next steps:")
    logger.info("   1. Edit train.py and optimize the inner_steps function")
    logger.info("   2. Test locally: uv run python -m tournament.test_local train.py")
    logger.info("   3. Submit when ready: uv run python -m neurons.miner train.py ...")


if __name__ == "__main__":
    main()
