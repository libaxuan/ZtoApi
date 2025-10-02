# Build stage
FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o main .

# Final stage
FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /app/main .

# Labels
LABEL maintainer="ZtoApi Contributors"
LABEL description="ZAI to GLM-4.6 API"
LABEL version="1.0.0"

# Expose port
EXPOSE 9090

# Run the application
CMD ["./main"]
