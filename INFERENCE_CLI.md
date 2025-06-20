# Digital Leadership Inference CLI

A command-line interface for inferring Digital Leadership archetypes from text sentences using the quantize_culture inference engine.

## Quick Start

```bash
# Basic usage - classify a sentence
python inference.py -s "order is the key for success" -c extended_contextualized_collection

# Get detailed analysis with all cluster members
python inference.py -s "order is the key for success" -c extended_contextualized_collection --verbose

# Get JSON output for programmatic use
python inference.py -s "order is the key for success" -c extended_contextualized_collection --format json
```

## Features

### 🎯 **Classification Results**
- **Cluster ID**: The matched cluster number
- **Similarity Score**: Vector similarity confidence (0-1)
- **Classification**: Digital Leadership archetype classification
- **Confidence Level**: STRONG, WEAK, AMBIGUOUS, TRAINING_MATCH, or NO_MATCH

### 🧠 **Dominant Logic Analysis** (Verbose Mode)
When using `--verbose`, the CLI provides:
- **Primary Category**: Most common DL category in the matched cluster
- **Primary Subcategory**: Most common DL subcategory
- **Primary Archetype**: Most common DL archetype
- **Distribution**: Complete breakdown of DL elements across cluster members

### 📚 **Cluster Analysis** (Verbose Mode)
- **All Cluster Members**: Every sentence in the matched cluster
- **DL Metadata**: Digital Leadership labels for each sentence
- **Cluster Size**: Total number of sentences in the cluster

## Command Options

| Option | Short | Description |
|--------|-------|-------------|
| `--sentence` | `-s` | The sentence to classify (required) |
| `--collection` | `-c` | Qdrant collection name to use as reference (required) |
| `--format` | | Output format: `human` (default) or `json` |
| `--verbose` | `-v` | Include detailed cluster analysis and dominant logic |
| `--help` | `-h` | Show help message |
| `--version` | | Show version information |

## Output Formats

### Human-Readable (Default)
```
📝 Sentence: "order is the key for success"
🎯 Digital Leadership Classification: cluster_46
📊 Cluster ID: 46
🔍 Similarity Score: 0.9281
✅ Confidence Level: AMBIGUOUS

🧠 DOMINANT LOGIC ANALYSIS:
   📂 Primary Category: Strategic Planning (5/8 sentences)
   📋 Primary Subcategory: Execution Focus (3/8 sentences)
   🎭 Primary Archetype: Results-Oriented (4/8 sentences)

📚 ALL CLUSTER MEMBERS (8 total):
    1. Domain Logic example phrase: Execution matters less than outcome. [Strategic Planning → Execution Focus → Results-Oriented]
    2. Planning is crucial for success [Strategic Planning → Process Design → Systematic]
    ...
```

### JSON Format
```json
{
  "sentence": "order is the key for success",
  "collection": "extended_contextualized_collection",
  "cluster_id": 46,
  "similarity_score": 0.9281,
  "classification": "cluster_46",
  "confidence_level": "AMBIGUOUS",
  "classification_status": "AMBIGUOUS_MATCH",
  "timestamp": "2025-06-20T17:08:09.065350",
  "dominant_logic": {
    "most_common_category": ["Strategic Planning", 5],
    "most_common_subcategory": ["Execution Focus", 3],
    "most_common_archetype": ["Results-Oriented", 4],
    "category_distribution": {"Strategic Planning": 5, "Innovation": 3},
    "subcategory_distribution": {"Execution Focus": 3, "Process Design": 2},
    "archetype_distribution": {"Results-Oriented": 4, "Systematic": 2}
  },
  "cluster_members": [...],
  "cluster_size": 8
}
```

## Examples

### Example 1: Basic Classification
```bash
python inference.py -s "innovation drives our success" -c extended_contextualized_collection
```

### Example 2: Detailed Analysis
```bash
python inference.py -s "fail fast and learn faster" -c extended_contextualized_collection --verbose
```

### Example 3: Programmatic Integration
```bash
python inference.py -s "order is everything" -c extended_contextualized_collection --format json | jq '.dominant_logic.most_common_category'
```

## Requirements

- **OpenAI API Key**: Set `OPENAI_API_KEY` environment variable
- **Qdrant Database**: Running Qdrant instance with populated collection
- **Python Dependencies**: Install via `pip install -r requirements.txt`

## Error Handling

The CLI provides clear error messages for common issues:
- Missing OpenAI API key
- Invalid collection names
- Network connectivity problems
- Malformed input sentences

Exit codes:
- `0`: Success
- `1`: Error occurred
- `130`: Interrupted by user (Ctrl+C)

## Integration

The CLI is designed for both interactive use and programmatic integration:

```python
import subprocess
import json

result = subprocess.run([
    'python', 'inference.py', 
    '-s', 'your sentence here',
    '-c', 'your_collection',
    '--format', 'json'
], capture_output=True, text=True)

if result.returncode == 0:
    data = json.loads(result.stdout)
    print(f"Classification: {data['classification']}")
    print(f"Confidence: {data['confidence_level']}")
```
