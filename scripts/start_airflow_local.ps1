$composeFile = "docker-compose.airflow.yml"

if (-not (Test-Path $composeFile)) {
    Write-Error "Compose file not found: $composeFile"
    exit 1
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

docker compose -f $composeFile up -d --build

if ($LASTEXITCODE -ne 0) {
    Write-Error "Airflow startup failed. Check Docker output above."
    exit $LASTEXITCODE
}

# Ensure admin user exists with known password.
docker compose -f $composeFile exec -T airflow-webserver airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
if ($LASTEXITCODE -ne 0) {
    docker compose -f $composeFile exec -T airflow-webserver airflow users reset-password --username admin --password admin
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Could not create/reset Airflow admin user automatically."
    }
}

Write-Host "Airflow is starting. Open http://localhost:8080"
Write-Host "Default user: admin"
Write-Host "Default password: admin"
