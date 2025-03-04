# triton-client-yolo
## OBB demo
0.进入triton docker

    0.1 docker run --gpus all --net=host --pid=host --ipc=host --privileged -it -v /your/local/dir:/home --name name nvcr.io/nvidia/tritonserver:24.08-py3
    
1.pip install -r requirements.txt

2.构造model_repository

    2.1 拿在本机构造的engine改名为model.plan放到目录trt_models/yolo11obb/1/里面(注意：此engine一定要是在你本机build出来的static batch size条件下的engine）
    
    2.2 拿在本机构造的engine改名为model.plan放到目录trt_dy_bs_models/yolo11obb/1/里面(注意：此engine一定要是在你本机build出来的dynamic batch size条件下engine）
    
3.启动tritonserver： LD_PRELOAD=/path/to/libcustom_plugins.so /opt/tritonserver/bin/tritonserver --model-repository=/path/to/trt_models

4.目前static_batch和single request脚本已经ready

    注意：single request和static batch共用一个config.pbtxt，dynamic batch的config.pbtxt不同，跑dynamic batch时需要重起另一个triton服务：LD_PRELOAD=/path/to/libcustom_plugins.so /opt/tritonserver/bin/tritonserver --model-repository=/path/to/trt_dy_bs_models
    
    4.0 single request运行命令 python3 single_request.py image /path/to/obb_images/ --model yolo11obb
    
    4.1 static batch运行命令  python3 static_batch_client.py image /path/to/obb_images/ --model yolo11obb
    
    4.2 dynamic batch运行命令 python3 dynamic_batch_client.py image /path/to/obb_images/ --model yolo11obb 
