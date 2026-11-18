"""Use the selected vLLM/NPU stack and its immutable baseline operator libraries."""
import ctypes,json,os
import torch
import torch_npu

for library in json.loads(os.environ.get('FULLNET_BASELINE_VENDOR_LIBS','[]')):
    ctypes.CDLL(library,mode=getattr(os,'RTLD_GLOBAL',256))

from vllm.entrypoints.openai import api_server

if __name__=='__main__':
    api_server.cli_env_setup()
    parser=api_server.make_arg_parser(api_server.FlexibleArgumentParser())
    args=parser.parse_args()
    args.async_scheduling=False
    args.enable_prefix_caching=False
    args.api_server_count=1
    api_server.validate_parsed_serve_args(args)
    api_server.uvloop.run(api_server.run_server(args))
