# Migration Guide: Old vs New Data Loading System

## What Changed?

The data loading system has been completely redesigned for better flexibility and extensibility.

## Old System (Removed)

```python
# Old way - rigid and limited
from src.data_loader import ToxicityDataLoader

loader = ToxicityDataLoader()
data = loader.load_combined_dataset(max_samples=1000)
```

**Problems with old system:**
- Hard-coded dataset list
- Single monolithic class
- Limited to Hugging Face datasets only
- Difficult to add new data sources
- No configuration-based approach

## New System (Current)

```python
# New way - flexible and extensible
from src.data_loaders import registry

# List available loaders
print(registry.list_loaders())

# Load from any source
loader = registry.get_loader("hate_speech")
data = loader.load(max_samples=1000)

# Or use command line
# python scripts/prepare_datasets.py --loader hate_speech --max-samples 1000
```

**Benefits of new system:**
- Plugin architecture - easily add new loaders
- Support for multiple data sources (HF, CSV, JSONL, API, etc.)
- Configuration-based data preparation
- Better error handling and logging
- Modular and testable code
- Command-line interface

## Migration Steps

### 1. Replace old imports
```python
# OLD
from src.data_loader import ToxicityDataLoader

# NEW
from src.data_loaders import registry, combiner
```

### 2. Update data loading code
```python
# OLD
loader = ToxicityDataLoader()
data = loader.load_hate_speech_offensive(max_samples=500)

# NEW
loader = registry.get_loader("hate_speech")
data = loader.load(max_samples=500)
```

### 3. Use configuration files for complex setups
```yaml
# configs/my_data_config.yaml
loaders:
  hate_speech:
    max_samples: 500
  csv:
    file_path: "my_custom_data.csv"
    max_samples: 300
```

```bash
python scripts/prepare_datasets.py --config configs/my_data_config.yaml
```

## Adding Custom Loaders

The new system makes it easy to add your own data sources:

```python
from src.data_loaders import registry, BaseDataLoader

class MyCustomLoader(BaseDataLoader):
    def load(self, **kwargs):
        # Your loading logic
        return samples
    
    def get_info(self):
        return {"name": "My Loader", "description": "..."}

# Register it
registry.register("my_loader", MyCustomLoader)

# Use it
python scripts/prepare_datasets.py --loader my_loader
```

## Backward Compatibility

The old `data/general_toxic_neutral_text_pairs.jsonl` format is still supported. The new system generates the same format, so existing vector computation scripts will work without changes.

## Need Help?

- Check `examples/custom_loader_example.py` for custom loader examples
- Use `--list-loaders` to see all available loaders
- Use `--loader-info <name>` to get detailed information about any loader
- Refer to `configs/flexible_data_config.yaml` for configuration examples