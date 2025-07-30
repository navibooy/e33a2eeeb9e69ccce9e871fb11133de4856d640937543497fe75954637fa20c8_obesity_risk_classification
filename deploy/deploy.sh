#!/bin/bash

# Deployment script for Obesity Classification ML Pipeline
# This script builds the Docker image and starts all services

set -e

echo "🚀 Starting deployment of Obesity Classification ML Pipeline..."

# Build the ML application Docker image
echo "📦 Building ML application Docker image..."
docker build -f deploy/docker/Dockerfile -t obesity-classification:latest .

# Create necessary directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p deploy/airflow/logs
mkdir -p deploy/airflow/config
mkdir -p deploy/airflow/plugins

# Start all services using docker-compose
echo "🐳 Starting services with docker-compose..."
cd deploy
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose ps

echo "✅ Deployment completed!"
echo "🌐 Airflow Web UI: http://localhost:8080"
echo "👤 Default credentials: airflow / airflow"
echo ""
echo "📋 Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
