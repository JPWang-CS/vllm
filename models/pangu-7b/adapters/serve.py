"""Pangu dense baseline using the deployed vLLM API-server entrypoint."""
from vllm.entrypoints.openai import api_server


if __name__ == "__main__":
    api_server.cli_env_setup()
    parser = api_server.make_arg_parser(api_server.FlexibleArgumentParser())
    args = parser.parse_args()
    # Explicit baseline behavior; these values are also saved in the run identity.
    args.async_scheduling = False
    args.enable_prefix_caching = False
    args.api_server_count = 1
    api_server.validate_parsed_serve_args(args)
    api_server.uvloop.run(api_server.run_server(args))
