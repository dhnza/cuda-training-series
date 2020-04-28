## **1. Matrix Row/Column Sums**

Your first task is to create a simple matrix row and column sum application in CUDA. The code skeleton is already given to you in *matrix_sums.cu*. Edit that file, paying attention to the FIXME locations, so that the output when run is like this:

```
row sums correct!
column sums correct!
```

After editing the code, compile it using the following:

```
module load cuda
nvcc -o matrix_sums matrix_sums.cu
```

The module load command selects a CUDA compiler for your use. The module load command only needs to be done once per session/login. *nvcc* is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++.

To run your code, we will use an LSF command:

```
bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1 ./matrix_sums
```

Alternatively, you may want to create an alias for your bsub command in order to make subsequent runs easier:

```
alias lsfrun='bsub -W 10 -nnodes 1 -P <allocation_ID> -Is jsrun -n1 -a1 -c1 -g1'
lsfrun ./matrix_sums
```

To run your code at NERSC on Cori, we can use Slurm:

```
module load esslurm
srun -C gpu -N 1 -n 1 -t 10 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10 ./matrix_sums
```

Allocation `m3502` is a custom allocation set up on Cori for this training series, and should be available to participants who registered in advance. If you cannot submit using this allocation, but already have access to another allocation that grants access to the Cori GPU nodes (such as m1759), you may use that instead.

If you prefer, you can instead reserve a GPU in an interactive session, and then run an executable any number of times while the Slurm allocation is active (this is recommended if there are enough available nodes):

```
salloc -C gpu -N 1 -t 60 -A m3502 --reservation cuda_training --gres=gpu:1 -c 10
srun -n 1 ./matrix_sums
```

Note that you only need to `module load esslurm` once per login session; this is what enables you to submit to the Cori GPU nodes.


If you have trouble, you can look at *matrix_sums_solution.cu* for a complete example.

## **2. Profiling**

We'll introduce something new: the profiler (in this case, Nsight Compute). We'll use the profiler first to time the kernel execution times, and then to gather some "metric" information that will possibly shed light on our observations.

It's necessary to complete task 1 first. Next, load the Nsight Compute module:
```
module load nsight-compute
```

Then, launch Nsight as follows:
(you may want to make your terminal session wide enough to make the output easy to read)

```
lsfrun nv-nsight-cu-cli ./matrix_sums
```

What does the output tell you?
Can you locate the lines that identify the kernel durations?
Are the kernel durations the same or different?
Would you expect them to be the same or different?


Next, launch *Nsight* as follows:

```
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sums
```

Our goal is to measure the global memory load efficiency of our kernels. In this case we have asked for two metrics: "*l1tex__t_sectors_pipe_lsu_mem_global_op_ld*" (the number of global memory load requests) and "*l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum*" (the number of sectors requested for global loads). This first metric above represents the denominator (requests) of the desired measurement (transactions per request) and the second metric represents the numerator (transactions). Dividing these numbers will give us the number of transactions per request. 

What similarities or differences do you notice between the *row_sum* and *column_sum* kernels?
Do the kernels (*row_sum*, *column_sum*) have the same or different efficiencies?
Why?
How does this correspond to the observed kernel execution times for the first profiling run?

Can we improve this?  (Stay tuned for the next CUDA training session.)

Here is a useful blog to help you get familiar with Nsight Compute: https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/


## **MY NOTES**

[This page](https://stackoverflow.com/questions/60535867/what-is-a-transaction-and-a-request-in-the-gld-transactions-per-request-metric)
explains the difference betwwen global loads and requests. Basically, a request is a warp-level
instruction and a transaction is a 32 byte load instruction. Since a float is 4 bytes and each
transaction is 32 bytes, a transaction loads 8 floats. Since each warp is loading 32 floats (1 per
thread), then each warp must make 32/8=4 transactions. Thus we have 4 transactions per request for a
perfectly colaesced memory access kernel like `column_sums`.

Also, if you don't have access to NSIGHT, here's the equivalent `nvprof` command:
```
nvprof -m global_load_requests,gld_transactions,gld_transactions_per_request,dram_read_throughput hw4/matrix_sums
```


