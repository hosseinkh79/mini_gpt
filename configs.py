
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model hyper_parameters
def get_gpt_configs():
    return dict(d_model = 12,
                vocab_size = 50257, #tokenizer.vocab_size
                seq_len = 100,
                num_encoders = 1,
                num_heads = 4,
                d_ff = 50,
                pos_drop = .3,
                encoder_drop = .5)

