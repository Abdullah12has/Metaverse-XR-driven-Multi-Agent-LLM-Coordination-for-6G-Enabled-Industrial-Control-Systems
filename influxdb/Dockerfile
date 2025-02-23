FROM influxdb:2.7

# Expose necessary ports
EXPOSE 8086

# Set up default InfluxDB variables
ENV DOCKER_INFLUXDB_INIT_MODE=setup \
    DOCKER_INFLUXDB_INIT_USERNAME=admin \
    DOCKER_INFLUXDB_INIT_PASSWORD=admin123 \
    DOCKER_INFLUXDB_INIT_ORG=my-org \
    DOCKER_INFLUXDB_INIT_BUCKET=my-bucket \
    DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-secret-token

# Start InfluxDB without the --config flag
CMD ["influxd"]
