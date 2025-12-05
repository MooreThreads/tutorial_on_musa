Matmul 自动化测试
# 1. 脚本说明
matmul 存放位置：
```shell
mudnn_bench
├── bench_test_matmul.sh
├── bin
│   ├── mudnn_bench -> mudnn_bench-x.x.x
│   └── mudnn_bench-x.x.x
├── matmul_test
```
mudnn_bench 示例：  
**部分旧版本mudnn_bench和mudnn版本不支持混合精度测试，需要和开发者做确认.**
```shell

# 示例 1：单卡，大矩阵，f32
MUSA_VISIBLE_DEVICES=4 ./bin/mudnn_bench -m --mm_m 6144 --mm_n 3584 --mm_k 6144 --warmup 30 --tm i --tmv 1000 -p -t f32

# 示例 2：多卡，标准尺寸，bf16
MUSA_VISIBLE_DEVICES=0,1 ./bin/mudnn_bench -m --mm_m 4096 --mm_n 4096 --mm_k 4096 --warmup 30 --tm i --tmv 1000 -p -t bf16

# 示例 3：单卡，特殊组合，int8
MUSA_VISIBLE_DEVICES=2 ./bin/mudnn_bench -m --mm_m 8192 --mm_n 8192 --mm_k 768 --warmup 30 --tm i --tmv 1000 -p -t int8

# 示例 4：使用混合精度格式
MUSA_VISIBLE_DEVICES=3 ./bin/mudnn_bench -m --mm_m 2048 --mm_n 2048 --mm_k 2048 --warmup 30 --tm i --tmv 1000 -p -t bf16:q4:bf16:bf16
```

# 2. 测试
可在测试脚本中自行批量配置测试MNK，warmup，iter等。
## 2.1 fp64, tf32 测试
注意：fp64和tf32 数据类型调用非 mudnn 接口
```shell
# 1. 编译
bash ./fp64_tf32_src/build_gemm_tf32.sh

bash ./fp64_tf32_src/build_gemm_fp64.sh

## 2. 测试
bash test_gemm_fp64_tf32.sh
```

## 2.2 f32_f16_bf16_q8_fp8 测试
mudnn_bench 测试矩阵value默认说明：
- 浮点：-0.5~0.5  
- fp8: 整型-10~10转浮点  
- qint4：-7～7 
- 整型：-127~127  
> 部分版本 mudnn_bench 工具支持全 0 测试(参数 `-z` 实现)，需要和开发者确认
```shell
bash test_gemm_f32_f16_bf16_q8_fp8.sh
```

## 2.3 混合精度测试
```shell
# A,B: fp16, C,D: f32: "f16:f16:f32:f32"
# A,B: bf16, C,D: f32: "bf16:bf16:f32:f32"
# A,B: tf32, C,D: f32: "f32"
# A,B: int8, C,D: int32: "int8"
# W8A8: "q8:q8:f32:f32"
# W4A16: "bf16:q4:bf16:bf16"
# A,B: fp8, C,D: fp16: "float8_e4m3:float8_e4m3:f16:f16"

bash test_gemm_mixed.sh
```

