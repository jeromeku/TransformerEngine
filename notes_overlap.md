See /home/jeromeku/transformerengine/transformer_engine/common/comm_gemm_overlap/comm_gemm_overlap.cpp
See comm_gemm_overlap bulk_overlap implementation for use of launchEvent for ordering comm launch -> compute launch

/home/jeromeku/transformerengine/transformer_engine/common/comm_gemm_overlap/userbuffers/userbuffers-host.cpp
See create_communicator_group for IPC handle exchange

See /home/jeromeku/transformerengine/transformer_engine/common/comm_gemm_overlap/userbuffers/userbuffers.cu for actual exchange kernels

/home/jeromeku/transformerengine/transformer_engine/pytorch/module/base.py for initialize_ub

Additional refs:
https://zhuanlan.zhihu.com/p/16594218518
https://blog.csdn.net/qq_27590277/article/details/144938140

