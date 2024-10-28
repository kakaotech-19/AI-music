FROM ubuntu:latest
LABEL authors="seminyang"

ENTRYPOINT ["top", "-b"]