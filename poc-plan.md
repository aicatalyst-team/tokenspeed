# PoC Plan: TokenSpeed

## Project Classification
- **Type:** model-serving
- **Key Technologies:** Python, PyTorch, CUDA, Triton Kernels, TensorRT-LLM integration, NCCL, FlashAttention, MLA (Multi-head Latent Attention), FastAPI/uvicorn (OpenAI-compatible HTTP server)
- **ODH Relevance:** TokenSpeed is a high-performance LLM inference engine that provides OpenAI-compatible model serving — a core use case for Open Data Hub / OpenShift AI. It can serve as an alternative to vLLM or TGI for production LLM deployments, particularly for agentic workloads on NVIDIA Blackwell/Hopper GPUs.

## PoC Objectives
What we want to prove:
1. **Containerization**: TokenSpeed can be built into a container image with all required GPU dependencies (CUDA, PyTorch, Triton, flash-attn, etc.)
2. **Model Loading**: The server can download and load a supported LLM (e.g., LLaMA 3.2 1B Instruct) from Hugging Face into GPU memory
3. **OpenAI-Compatible Serving**: The `/v1/chat/completions`, `/v1/completions`, and `/v1/models` endpoints work correctly and return well-formed responses
4. **Streaming**: Server-Sent Events (SSE) streaming works for chat completions
5. **Health Monitoring**: The `/health` endpoint reports server readiness and can be used as a Kubernetes readiness probe

## Infrastructure Requirements
- **Inference Server:** Custom (TokenSpeed is its own inference server with OpenAI-compatible API)
- **Vector Database:** none
- **Embedding Model:** none
- **GPU Required:** Yes — NVIDIA GPU with CUDA support (minimum: 1x GPU with ≥16 GB VRAM for a 1B parameter model; ideally A100/H100/B200)
- **Persistent Storage:** 50Gi PVC for model weight caching (Hugging Face Hub downloads)
- **Resource Profile:** gpu (8Gi RAM minimum, 4 CPU, 1 NVIDIA GPU)
- **Sidecar Containers:** none

## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (secret) | Hugging Face API token for gated model downloads |
| `TOKENSPEED_MODEL` | Yes | Model identifier, e.g. `meta-llama/Llama-3.2-1B-Instruct` |
| `TOKENSPEED_HOST` | No (default `0.0.0.0`) | Host to bind the HTTP server |
| `TOKENSPEED_PORT` | No (default `8000`) | Port to bind the HTTP server |
| `HF_HOME` | No | Override Hugging Face cache directory (useful for PVC mount) |
| `TOKENSPEED_TP_SIZE` | No | Tensor parallelism size (number of GPUs) |

## Test Scenarios

### Scenario 1: Health Check
- **Description:** Verify the TokenSpeed server has fully started and the model is loaded
- **Type:** http (GET)
- **Endpoint:** `/health`
- **Input:** none
- **Expected:** Returns HTTP 200 OK. The server exposes a `/health` endpoint from its HTTP server module.
- **Timeout:** 300 seconds (model download + loading can take several minutes on first start)
- **Notes:** This should be used as the Kubernetes readiness probe

### Scenario 2: List Models
- **Description:** Verify the OpenAI-compatible `/v1/models` endpoint returns the loaded model
- **Type:** http (GET)
- **Endpoint:** `/v1/models`
- **Input:** none
- **Expected:** Returns 200 with JSON body containing a `data` array with at least one entry whose `id` matches the loaded model name
- **Timeout:** 30 seconds

### Scenario 3: Chat Completion
- **Description:** Send a simple chat completion request to verify end-to-end inference
- **Type:** http (POST)
- **Endpoint:** `/v1/chat/completions`
- **Input:** `{"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "What is 2 + 2? Answer in one word."}], "max_tokens": 16, "temperature": 0.0}`
- **Expected:** Returns 200 with JSON containing `choices[0].message.content` that is non-empty
- **Timeout:** 60 seconds

### Scenario 4: Text Completion
- **Description:** Verify the legacy `/v1/completions` endpoint works for text generation
- **Type:** http (POST)
- **Endpoint:** `/v1/completions`
- **Input:** `{"model": "meta-llama/Llama-3.2-1B-Instruct", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0.0}`
- **Expected:** Returns 200 with JSON containing generated text mentioning "Paris"
- **Timeout:** 60 seconds

### Scenario 5: Streaming Chat Completion
- **Description:** Verify that SSE streaming works for chat completions
- **Type:** http (POST)
- **Endpoint:** `/v1/chat/completions`
- **Input:** `{"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 16, "temperature": 0.0, "stream": true}`
- **Expected:** Returns 200 with `Content-Type: text/event-stream`, SSE events containing `delta` content chunks, stream ends with `data: [DONE]`
- **Timeout:** 60 seconds

## Dockerfile Considerations

TokenSpeed is a complex GPU-accelerated Python application. The Dockerfile must handle:

1. **Base Image**: Use an NVIDIA CUDA base image (e.g., `nvidia/cuda:12.8.1-devel-ubuntu22.04` or the PyTorch NGC container `nvcr.io/nvidia/pytorch:24.xx-py3`). The existing `docker/Dockerfile` in the repo should be referenced for build guidance but may need adaptation for OpenShift (non-root user, etc.).

2. **Python Dependencies**: Install from `python/pyproject.toml` using pip. The project has extensive GPU dependencies including:
   - PyTorch with CUDA support
   - Triton
   - flash-attn / flashinfer
   - Various CUDA kernel extensions
   - FastAPI, uvicorn, aiohttp for the HTTP server

3. **Entry Point**: The server is launched via the CLI module:
   ```
   python -m tokenspeed.cli serve --model $TOKENSPEED_MODEL --host 0.0.0.0 --port 8000
   ```
   The CLI module (`python/tokenspeed/cli.py`) dispatches to `tokenspeed.api_server` which calls `launch_server()` from the HTTP entrypoints module.

4. **Port**: EXPOSE 8000 — the server listens on port 8000 by default.

5. **Working Directory**: Set to `/workspace` or `/app`. The Python package is in the `python/` subdirectory.

6. **Non-root User**: OpenShift requires running as non-root. Add a non-root user and ensure the HF cache directory is writable.

7. **Model Cache Volume**: The container should mount a PVC at the HF cache directory (e.g., `/home/user/.cache/huggingface`) so model weights persist across restarts.

8. **Build Complexity Note**: This project has many optional C/CUDA extensions. For the PoC, focus on getting the core Python package installed. Some kernel backends may fail to compile — that's acceptable as long as the server starts with at least one working attention backend.

## Deployment Considerations

1. **Deployment Model**: Deploy as a Kubernetes **Deployment** with 1 replica. This is a long-running HTTP server.

2. **Service**: Create a ClusterIP Service on port 8000 targeting the pod's port 8000.

3. **GPU Resources**: The pod spec MUST request `nvidia.com/gpu: 1` in resource limits. Without a GPU, the server cannot load any model.

4. **Readiness Probe**: Use HTTP GET `/health` on port 8000 with `initialDelaySeconds: 120` and `periodSeconds: 10`. Model loading takes significant time.

5. **Liveness Probe**: Use HTTP GET `/health` on port 8000 with `initialDelaySeconds: 180` and `periodSeconds: 30`.

6. **PVC**: Mount a 50Gi PVC at `/home/user/.cache/huggingface` (or wherever `HF_HOME` points) for model weight caching. First-time download of even a 1B model takes several GB.

7. **Secrets**: Create a Kubernetes Secret for `HF_TOKEN` and mount it as an environment variable.

8. **Resource Requests/Limits**:
   - CPU: 4 cores request, 8 cores limit
   - Memory: 8Gi request, 16Gi limit
   - GPU: 1 nvidia.com/gpu

9. **Security Context**: Run as non-root (UID 1001). Ensure writable tmp and cache directories.

10. **Test Strategy**: HTTP — all test scenarios send HTTP requests to the Service endpoint. Wait for the readiness probe to pass before running tests. The health check has a 300-second timeout to accommodate model download and loading.

11. **Model Selection for PoC**: Use `meta-llama/Llama-3.2-1B-Instruct` as the smallest viable model for proving the engine works. This requires ~2-3 GB VRAM and downloads quickly. For production benchmarking, larger models (DeepSeek V3, Kimi K2.5) would be used but require multi-GPU setups.