data_dir=DATADIR
log_dir=LOGDIR
baseline_name="mask2former"

SHELL = /bin/sh

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

build:
	docker build -t $(baseline_name)_docker -f Dockerfile .
train_plants:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(CURDIR)/../phenorob_challenge_tools:/src/phenorob_challenge_tools \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	-e DETECTRON2_DATASETS="/data" \
        -e CUDA_VISIBLE_DEVICES=0 \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=40960m \
	$(baseline_name)_docker \
	python train_net.py --num-gpus 1 --config-file /configs/maskformer_plants.yaml

train_leaves:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(CURDIR)/../phenorob_challenge_tools:/src/phenorob_challenge_tools \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	-e DETECTRON2_DATASETS="/data" \
        -e CUDA_VISIBLE_DEVICES=0 \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=40960m \
	$(baseline_name)_docker \
	python train_net.py --num-gpus 1 --config-file /configs/maskformer_leaves.yaml

predict_test_plants:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(CURDIR)/../phenorob_challenge_tools:/src/phenorob_challenge_tools \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	-e DETECTRON2_DATASETS="/data" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=2048m \
	$(baseline_name)_docker \
	python predict.py \
	--cfg-file /configs/maskformer_plants.yaml \
	--model-weights /logs/mask2former_plants/model_best.pth \
	--testset-folder /data/test/images \
	--output-folder /logs/mask2former_plants/test_predictions

predict_val_plants:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(CURDIR)/../phenorob_challenge_tools:/src/phenorob_challenge_tools \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	-e DETECTRON2_DATASETS="/data" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=2048m \
	$(baseline_name)_docker \
	python predict.py \
	--cfg-file /configs/maskformer_plants.yaml \
	--model-weights /logs/mask2former_plants/model_best.pth \
	--testset-folder /data/val/images \
	--output-folder /logs/mask2former_plants/val_predictions

predict_test_leaves:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(CURDIR)/../phenorob_challenge_tools:/src/phenorob_challenge_tools \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	-e DETECTRON2_DATASETS="/data" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=2048m \
	$(baseline_name)_docker \
	python predict.py \
	--cfg-file /configs/maskformer_leaves.yaml \
	--model-weights /logs/mask2former_leaves/model_best.pth \
	--testset-folder /data/test/images \
	--output-folder /logs/mask2former_leaves/test_predictions

predict_val_leaves:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(CURDIR)/../phenorob_challenge_tools:/src/phenorob_challenge_tools \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	-e DETECTRON2_DATASETS="/data" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=2048m \
	$(baseline_name)_docker \
	python predict.py \
	--cfg-file /configs/maskformer_leaves.yaml \
	--model-weights /logs/mask2former_leaves/model_best.pth \
	--testset-folder /data/val/images \
	--output-folder /logs/mask2former_leaves/val_predictions

evaluate_plants:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(CURDIR)/../evaluation_scripts:/src/evaluation_scripts \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	-e DETECTRON2_DATASETS="/data" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=1024m \
	$(baseline_name)_docker \
	python ../evaluation_scripts/evaluate_plant_instance_masks_panoptic.py --ground_truth_dir "/data/test" --prediction_dir "/logs/mask2former_plants/test_predictions"
test:
	docker run -it \
	--user $(CURRENT_UID):$(CURRENT_GID) \
	-v $(CURDIR)/src:/src \
	-v $(CURDIR)/../evaluation_scripts:/src/evaluation_scripts \
	-v $(data_dir):/data \
	-v $(log_dir):/logs \
	-v $(CURDIR)/configs:/configs \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY \
	-e TORCH_HOME="/logs" \
	-e DETECTRON2_DATASETS="/data" \
	--gpus 'all,"capabilities=compute,utility,graphics"' \
	--shm-size=1024m \
	$(baseline_name)_docker \
	bash