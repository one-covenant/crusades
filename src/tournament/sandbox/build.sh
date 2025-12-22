#!/bin/bash
set -euo pipefail

# Build and push the tournament sandbox image to GHCR
#
# Usage:
#   ./build.sh [--push] [--tag TAG]
#
# Prerequisites:
#   - Docker installed and running
#   - For push: authenticated to ghcr.io (gh auth login or docker login ghcr.io)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="ghcr.io/one-covenant/templar-tournament-sandbox"
IMAGE_TAG="latest"
PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================================"
echo "Building Tournament Sandbox Image"
echo "============================================================"
echo "Image: ${FULL_IMAGE}"
echo "Context: ${SCRIPT_DIR}"
echo ""

cd "${SCRIPT_DIR}"

docker build \
    --platform linux/amd64 \
    -t "${FULL_IMAGE}" \
    -f Dockerfile \
    .

echo ""
echo "Build complete: ${FULL_IMAGE}"

if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing to registry..."
    docker push "${FULL_IMAGE}"
    echo "Push complete: ${FULL_IMAGE}"
fi

echo ""
echo "============================================================"
echo "Done"
echo "============================================================"
echo ""
echo "To run locally:"
echo "  docker run --gpus all -p 8000:8000 ${FULL_IMAGE}"
echo ""
echo "To push to registry:"
echo "  docker push ${FULL_IMAGE}"
echo ""
echo "To deploy to Basilica:"
echo "  python deploy.py"
