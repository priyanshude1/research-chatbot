# Base image: an official Python 3.11 build on Debian, "-slim" variant
# (stripped of docs/compilers/etc. to stay small). We don't need a GPU base
# image since the LLM runs remotely via generator.py's provider APIs — the
# only local model is the ~90MB embedding model in embedder.py, which runs
# fine on CPU.
FROM python:3.11-slim

# All subsequent instructions run from /app inside the image. Created
# automatically by WORKDIR if it doesn't exist. Every relative COPY/RUN
# below is resolved against this.
WORKDIR /app

# Copy ONLY requirements.txt first, then install, BEFORE copying the rest
# of the source code. This is a deliberate layer-ordering trick: Docker
# caches each instruction's result keyed on its inputs. As long as
# requirements.txt doesn't change, this pip install layer is reused from
# cache on every rebuild, even if every .py file changed — so editing
# src/agent.py and rebuilding takes seconds, not the minutes a full
# dependency reinstall (torch, sentence-transformers, chromadb) would take.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers embedding model (embedder.py's
# all-MiniLM-L6-v2) into the image at BUILD time, not first request. Without
# this, the first container start would need outbound internet access to
# Hugging Face just to answer its first query — an easy-to-miss dependency
# that would silently fail on a network-restricted host, and adds latency
# on top of Render's free-tier cold start. Baking it in trades a larger
# image for a container that's ready immediately and works offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Now copy everything else (src/, api/, index.py, data/, CLAUDE.md, etc.)
# per .dockerignore's exclusions. This layer invalidates on almost every
# rebuild (any source change), which is exactly why it's placed last —
# nothing expensive depends on it.
COPY . .

# Documentation only — EXPOSE does not actually publish the port. It's a
# note (to humans and to tools like `docker inspect`) that the container
# listens on 8000. The actual host<->container port mapping happens via
# `docker run -p` or docker-compose.yml's "ports:", not here.
EXPOSE 8000

# The command that runs when a container starts from this image.
# --host 0.0.0.0 is required (not optional) inside a container: uvicorn's
# default 127.0.0.1 only accepts connections from inside the same network
# namespace, which would make the app unreachable from outside the
# container even with a port mapping.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
