tag=nvcr.io/nvidia/pytorch:23.04-py3_sdpa
# build performance environment
docker build -f Dockerfile -t $tag .

# performance test
for num_heads in 16 32; do
    for batch_size in 1 2 4 8 16 32 64 128; do
        for seq in 768 1024 2048 4096 8192; do
        docker run  --gpus all -v ${HOME}/Repo/model-optimizer/test_sdpa:/workspace $tag \
            python3 -u test_sdpa.py --batch_size $batch_size --seq_len ${seq} --num_heads ${num_heads}
        done
    done
done

for num_heads in 20 40; do
    for batch_size in 1 2 4 8 16 32 64 128; do
        for seq in 1920; do
        docker run  --gpus all -v ${HOME}/Repo/model-optimizer/test_sdpa:/workspace $tag \
            python3 -u test_sdpa.py --batch_size $batch_size --seq_len ${seq} --num_heads ${num_heads}
        done
    done
done
for num_heads in 112; do
    for batch_size in 1 2 4 8 16 32 64 128; do
        for seq in 14336; do
        docker run  --gpus all -v ${HOME}/Repo/model-optimizer/test_sdpa:/workspace $tag \
            python3 -u test_sdpa.py --batch_size $batch_size --seq_len ${seq} --num_heads ${num_heads}
        done
    done
done

