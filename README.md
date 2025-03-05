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

## log of fake quant using TensorRT modelOpt

```
root@machine:/home/TensorRT-Model-Optimizer/examples/onnx_ptq# python -m modelopt.onnx.quantization \
    --onnx_path=./yolo11n-obb.onnx \
    --trt_plugins=/home/chapter3/lib/plugin/libcustom_plugins.so
INFO:root:No output path specified, save quantized model to ./yolo11n-obb-fake.quant.onnx
INFO:root:Model with ORT support is saved to ./yolo11n-obb_ort_support.onnx. Model contains custom ops: ['EfficientRotatedNMS_TRT'].
INFO:root:Model ./yolo11n-obb_ort_support.onnx with opset_version 11 is loaded.
INFO:root:Model is cloned to ./yolo11n-obb_opset13.onnx with opset_version 13.
INFO:root:Model is cloned to ./yolo11n-obb_named.onnx after naming the nodes.
INFO:root:Successfully imported the `tensorrt` python package with version 10.8.0.43.
INFO:root:libcudnn*.so* is accessible in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Quantization Mode: int8
INFO:root:Successfully imported the `tensorrt` python package with version 10.8.0.43.
INFO:root:libcudnn*.so* is accessible in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Quantizable op types in the model: ['Resize', 'Add', 'Mul', 'MaxPool', 'MatMul', 'Conv']
INFO:root:Building non-residual Add input map ...
INFO:root:Searching for hard-coded patterns like MHA, LayerNorm, etc. to avoid quantization.
INFO:root:Building KGEN/CASK targeted partitions ...
INFO:root:Classifying the partition nodes ...
INFO:root:Successfully imported the `tensorrt` python package with version 10.8.0.43.
INFO:root:libcudnn*.so* is accessible in /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
INFO:root:Total number of nodes: 491
WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
Collecting tensor data and making histogram ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 116/116 [00:00<00:00, 385.00it/s]
Finding optimal threshold for each tensor using 'entropy' algorithm ...
Number of tensors : 116
Number of histogram bins : 128 (The number may increase depends on the data it collects)
Number of quantized bins : 128
WARNING:root:Please consider pre-processing before quantization. See https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
INFO:root:Deleting QDQ nodes from marked inputs to make certain operations fusible ...
INFO:root:Total number of quantized nodes: 126
INFO:root:Quantized type counts: {'Conv': 97, 'Add': 15, 'MaxPool': 3, 'Concat': 5, 'Reshape': 1, 'MatMul': 2, 'Resize': 2, 'Sub': 1}
INFO:root:Quantized onnx model is saved as ./yolo11n-obb-fake.quant.onnx
```
