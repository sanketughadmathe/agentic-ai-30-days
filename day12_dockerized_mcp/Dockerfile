FROM python:3.11-slim

# ---- security hygiene ----
RUN useradd -m mcpuser
WORKDIR /app

# ---- dependencies ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- app ----
COPY mcp_server.py /app/mcp_server.py

# ---- drop root ----
USER mcpuser

EXPOSE 3333

CMD ["python", "mcp_server.py"]
