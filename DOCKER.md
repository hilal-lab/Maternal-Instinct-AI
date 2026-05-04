# Maternal Instinct AI - Docker Setup

This Docker setup allows you to run the Maternal Instinct AI application with all its dependencies in containers.

## Services

1. **Ollama** - Local LLM server (port 11434)
2. **Backend** - FastAPI server providing AI services (port 8000)
3. **Frontend** - Next.js application serving the user interface (port 3000)

## Prerequisites

- Docker Engine
- Docker Compose v2+

## Quick Start

1. Navigate to the project directory:
   ```bash
   cd /home/lerainfrog/Documents/PROF-RIRI/Maternal\ AI/Maternal-Instinct-AI/
   ```

2. Start all services:
   ```bash
   docker compose up -d
   ```

3. Wait for services to initialize and check status:
   ```bash
   docker compose ps
   ```

4. Verify services are running:
   - Ollama API: http://localhost:11434
   - Backend API: http://localhost:8000
   - Frontend UI: http://localhost:3000

## Installing Models

After starting the services, you need to pull the required AI models. Use the management script:

```bash
# Pull all required models
./scripts/manage_models.sh pull-required

# Or pull individually
./scripts/manage_models.sh pull llama3.1:8b
./scripts/manage_models.sh pull bge-m3

# List installed models
./scripts/manage_models.sh list
```

## Common Commands

### Start/Stop Services
```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

### Viewing Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f ollama
docker compose logs -f backend
docker compose logs -f frontend
```

### Rebuilding Containers
```bash
# Rebuild after code changes
docker compose up -d --build

# Rebuild without cache
docker compose build --no-cache
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama server URL |
| `PYTHONPATH` | `/app` | Python module search path |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API URL for frontend |

## Container Health

All services have health checks configured:
- **Ollama**: TCP port check on 11434
- **Backend**: HTTP check on /
- **Frontend**: HTTP check on /

Wait for all services to show `(healthy)` status before testing.

## Troubleshooting

### Ollama container fails to start
```bash
# Check if port 11434 is already in use
lsof -i :11434

# View Ollama logs
docker compose logs ollama
```

### Backend import errors
```bash
# View backend logs
docker compose logs backend

# Rebuild backend
docker compose up -d --build backend
```

### Frontend not loading
```bash
# Check frontend logs
docker compose logs frontend

# Rebuild frontend
docker compose up -d --build frontend
```

## Production Deployment Notes

For production, consider:
1. Removing volume mounts for read-only source code
2. Using a reverse proxy (nginx) for HTTPS
3. Setting proper CORS origins
4. Using persistent storage for database and models
5. Implementing proper logging and monitoring