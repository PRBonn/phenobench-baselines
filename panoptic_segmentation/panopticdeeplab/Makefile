data_dir=DATADIR
log_dir=LOGDIR
baseline_name="deeplab"

SHELL = /bin/sh

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

build:
	docker build -t $(baseline_name)_docker -f Dockerfile .
train_plants:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
        -e CUDA_VISIBLE_DEVICES=0 \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=256m \
	$(baseline_name)_docker \
	python tools/train_net.py --cfg "/configs/panoptic_deeplab_plants.yaml" --inst_type "plants"

train_leaves:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=256m \
	$(baseline_name)_docker \
	python tools/train_net.py --cfg "/configs/panoptic_deeplab_leaves.yaml"  --inst_type "leaves"

predict_test_plants:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=256m \
	$(baseline_name)_docker \
	python tools/predict.py --cfg "/configs/panoptic_deeplab_plants.yaml" --inst_type "plants" --data_set "test"

predict_val_plants:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=256m \
	$(baseline_name)_docker \
	python tools/predict.py --cfg "/configs/panoptic_deeplab_plants.yaml" --inst_type "plants" --data_set "val"

predict_leaves:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=256m \
	$(baseline_name)_docker \
	python tools/predict.py --cfg "/configs/panoptic_deeplab_leaves.yaml" --inst_type "leaves"

predict_plants_shit:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=256m \
	$(baseline_name)_docker \
	python tools/demo.py --cfg "/configs/panoptic_deeplab_plants.yaml" \
	--input-files /data/test/images \
	--output-dir /logs/pandeeplab_plants/test_predictions


evaluate:
	docker run -it -v $(CURDIR)/deeplab_baseline:/baseline deeplab_baseline_docker python3 test.py
test:
	echo $(CURRENT_UID)
