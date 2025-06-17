# Use the official Qdrant DB Docker image
FROM qdrant/qdrant:latest

# Expose default Qdrant port
EXPOSE 6333

# Set up persistent storage
VOLUME ["/qdrant/storage"]

# Start Qdrant DB
CMD ["/qdrant/qdrant"]
