# FROM python:3.11-slim

# RUN useradd -m mcpuser
# WORKDIR /app

# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# COPY mcp_server_limited.py /app/mcp_server_limited.py

# USER mcpuser
# EXPOSE 3333

# CMD ["python", "mcp_server_limited.py"]

FROM python:3.11-slim

# -------------------------
# Security: non-root user
# -------------------------
RUN useradd -m -u 1000 mcpuser

WORKDIR /app

# -------------------------
# Dependencies
# -------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Application
# -------------------------
COPY mcp_server_limited.py /app/mcp_server_limited.py

# -------------------------
# Drop privileges
# -------------------------
USER mcpuser

EXPOSE 3333

# -------------------------
# Resource-aware startup
# -------------------------
# Note: Actual resource limits are set at runtime with docker run flags:
# --memory="128m" --cpus="0.5" --pids-limit=50
CMD ["python", "mcp_server_limited.py"]
