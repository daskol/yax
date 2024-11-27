IMAGE = ghcr.io/daskol/yax
TAG = 2024.11
REV = 1

all:

build-image:
	docker build -t $(IMAGE):$(TAG)-$(REV) .
	docker tag $(IMAGE):$(TAG)-$(REV) $(IMAGE):latest
