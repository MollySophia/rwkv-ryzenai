import torch

def extract_info_from_model_cfg(model_cfg):
    return model_cfg.n_layer, model_cfg.n_head, model_cfg.n_embd

def get_dummy_state(batch_size, model_cfg, device):
    def _cache(shape, fp16):
        if fp16:
            return torch.zeros(shape, dtype=torch.float16).to(device=device)
        else:
            return torch.zeros(shape).to(device=device)

    num_layers, num_heads, embed_dim = extract_info_from_model_cfg(model_cfg)
    head_size = embed_dim // num_heads

    state_0 = (batch_size, embed_dim)
    state_1 = (batch_size, num_heads, head_size, head_size)
    state_2 = (batch_size, embed_dim)
 
    state = []
    for _ in range(0, num_layers):
        state += [_cache(state_0, model_cfg.fp16), _cache(state_1, model_cfg.fp16), _cache(state_2, model_cfg.fp16)]
    return state

def get_dummy_input_for_rwkv_causal_llm(batch_size, input_length, device, model_cfg=None):
    input_ids = torch.LongTensor([[0]*input_length for _ in range(batch_size)]).to(device)
    inputs = {'in0': input_ids, 'state': get_dummy_state(batch_size, model_cfg, device)}
    return inputs

def to_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.detach().clone().to(device)
    if isinstance(t, tuple):
        return tuple([to_device(i, device) for i in t])
    if isinstance(t, list):
        return [to_device(i, device) for i in t]
    if isinstance(t, dict):
        return {k:to_device(v, device) for k,v in t.items()}
    return t

def to_cpu(t):
    return to_device(t, torch.device('cpu'))

def get_input_output_names(model_cfg):
    num_layers, _, _ = extract_info_from_model_cfg(model_cfg)
    def _get_state_names(sfx, n_layers):
        all = []
        for i in range(n_layers):
            for j in range(0, 3):
                all.extend([f'layer{i}_state{j}_{sfx}'])
        return all

    input_names = ['input_ids']
    input_names += _get_state_names('in', num_layers)
    output_names = ['logits']
    output_names += _get_state_names('out', num_layers)
    return input_names, output_names
