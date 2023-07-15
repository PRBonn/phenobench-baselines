data_dir=${PARENT_DIR_PATH}
baseline_name=yolov7

SHELL = /bin/sh

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

build:
	docker build -t $(baseline_name)_docker -f Dockerfile --build-arg USER_ID=$(CURRENT_UID) --build-arg GROUP_ID=$(CURRENT_GID) .

train:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
	-v ${data_dir}/YOLO_format:/data \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/train.py --workers 8 --batch-size 16 --data /src/data/phenobench_plants.yaml --img 1024 1024 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

val:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
        -v ${data_dir}:/data \
        -v ${IN_IMGS_DIR}:/in_imgs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
        -e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/detect.py --img-size 1024 --conf-thres 0.0 --iou-thres 0.65 --weights /src/weights/yolov7_plant_detection.pt --save-txt --save-conf --source /in_imgs

infer:
	docker run \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
        -v ${data_dir}:/data \
        -v ${IN_IMGS_DIR}:/in_imgs \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
        -e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
	python yolov7/detect.py --img-size 1024 --conf-thres 0.25 --iou-thres 0.65 --weights /src/weights/yolov7_plant_detection.pt --save-txt --save-conf --source /in_imgs

bash:
	docker run -it \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
        -v $(CURDIR)/scripts:/scripts \
	-v ${data_dir}:/data \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
	-e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
				bash 

convert_annos2yolo:
	docker run -it \
        --user $(CURRENT_UID):$(CURRENT_GID) \
        -v $(CURDIR)/src:/src \
        -v $(CURDIR)/scripts:/scripts \
        -v $(data_dir):/data \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
        -e CUDA_VISIBLE_DEVICES \
        -e TORCH_HOME="/logs" \
        --gpus 'all,"capabilities=compute,utility,graphics"' \
        --shm-size=64g \
        $(baseline_name)_docker \
        python /scripts/convert_annos2yolo.py --input_dir /data --output_dir /data/YOLO_format

