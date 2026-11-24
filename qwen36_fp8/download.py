import sys
from huggingface_hub import snapshot_download
print(snapshot_download('Logics-MLLM/Logics-SWE-Qwen3.6-27B',revision=sys.argv[1],local_dir=sys.argv[2],max_workers=8,
  allow_patterns=['*.safetensors','model.safetensors.index.json','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','processor_config.json']))
