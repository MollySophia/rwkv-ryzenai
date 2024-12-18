import os
import sys
import subprocess
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
from rwkv_src.rwkv_tokenizer import RWKV_TOKENIZER
from utils.model_utils import init_inputs_for_rwkv_onnx, sample_logits

def get_apu_info():
    # Run pnputil as a subprocess to enumerate PCI devices
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Check for supported Hardware IDs
    apu_type = ''
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(errors="ignore"): apu_type = 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(errors="ignore"): apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(errors="ignore"): apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(errors="ignore"): apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in stdout.decode(errors="ignore"): apu_type = 'KRK'
    return apu_type

def set_environment_variable(apu_type):
    install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
    match apu_type:
        case 'PHX/HPT':
            print("Setting environment for PHX/HPT")
            os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '1x4.xclbin')
            os.environ['NUM_OF_DPU_RUNNERS']='1'
            os.environ['XLNX_TARGET_NAME']='AMD_AIE2_Nx4_Overlay'
        case ('STX' | 'KRK'):
            print("Setting environment for STX")
            os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_Nx4_Overlay.xclbin')
            os.environ['NUM_OF_DPU_RUNNERS']='1'
            os.environ['XLNX_TARGET_NAME']='AMD_AIE2_Nx4_Overlay'
        case _:
            print("Unrecognized APU type. Exiting.")
            exit()
    print('XLNX_VART_FIRMWARE=', os.environ['XLNX_VART_FIRMWARE'])
    print('NUM_OF_DPU_RUNNERS=', os.environ['NUM_OF_DPU_RUNNERS'])
    print('XLNX_TARGET_NAME=', os.environ['XLNX_TARGET_NAME'])

parser = argparse.ArgumentParser(description='Test onnx model')
parser.add_argument('model', type=Path, help='Path to RWKV onnx file')
parser.add_argument('--use_ryzenai', action='store_true', help='Use RyzenAI')

parser_args = parser.parse_args()

session = None
if parser_args.use_ryzenai:
    # Get APU type info: PHX/STX/HPT
    apu_type = get_apu_info()

    # set environment variables: XLNX_VART_FIRMWARE and NUM_OF_DPU_RUNNERS
    set_environment_variable(apu_type)

    install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
    config_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json')
    cache_key   = 'modelcachekey_rwkvtest'
    providers   = ['VitisAIExecutionProvider']

    provider_options = [{
                'config_file': config_file,
                'cacheKey': cache_key,
        }]
    session = ort.InferenceSession(
        str(parser_args.model),
        providers=providers,
        provider_options=provider_options
    )
else:
    session = ort.InferenceSession(str(parser_args.model))

tokenizer = RWKV_TOKENIZER('assets/rwkv_vocab_v20230424.txt')
prompt = "User: Hello!\n\nAssistant:"
prompt_ids = tokenizer.encode(prompt)

n_embed = session.get_inputs()[1].shape[-1]
n_head = session.get_inputs()[2].shape[-3]
head_size = session.get_inputs()[2].shape[-1]
n_layer = (len(session.get_inputs()) - 1) // 3
assert n_embed == n_head * head_size

print(f"n_head: {n_head}, head_size: {head_size}, n_layer: {n_layer}")

inputs = init_inputs_for_rwkv_onnx(n_head, head_size, n_layer, 1, 1, dtype=np.float16)

logits = None
for id in prompt_ids:
    inputs['input_ids'][0, 0] = id
    outputs = session.run(None, inputs)
    logits = outputs[0][0, 0]
    for i in range(n_layer):
        inputs[f'layer{i}_state0_in'] = outputs[3*i+1]
        inputs[f'layer{i}_state1_in'] = outputs[3*i+2]
        inputs[f'layer{i}_state2_in'] = outputs[3*i+3]

GEN_LENGTH = 100
print(prompt, end='', flush=True)
for _ in range(GEN_LENGTH):
    token = sample_logits(logits)
    print(tokenizer.decode([token]), end='', flush=True)
    inputs['input_ids'][0, 0] = token
    outputs = session.run(None, inputs)
    logits = outputs[0][0, 0]
    for i in range(n_layer):
        inputs[f'layer{i}_state0_in'] = outputs[3*i+1]
        inputs[f'layer{i}_state1_in'] = outputs[3*i+2]
        inputs[f'layer{i}_state2_in'] = outputs[3*i+3]
