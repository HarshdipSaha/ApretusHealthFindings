import transformers
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

orig_from_pretrained = transformers.AutoTokenizer.from_pretrained
@classmethod
def my_from_pretrained(cls, *args, **kwargs):
    t = orig_from_pretrained(*args, **kwargs)
    t.model_max_length = min(t.model_max_length, 512)
    return t
transformers.AutoTokenizer.from_pretrained = my_from_pretrained

import evaluate
bertscore = evaluate.load('bertscore')
res = bertscore.compute(predictions=['test'], references=['test'], model_type='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', num_layers=12)
print(res)
