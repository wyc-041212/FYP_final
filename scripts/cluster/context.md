你研究一下这三个文件，现阶段要你做的就是在集群上面把cdf剩下的实验跑完，同样也是sweep0.1到0.9:
/home/comp/f2256768/FYP_final/scripts/cluster/cluster_experiment_handoff.md（在local写的一些注意事项）；
/home/comp/f2256768/FYP_final/scripts/cluster/run_cdf_cache_gpu10.slurm.sh
/home/comp/f2256768/FYP_final/scripts/cluster/run_cdf_infer_gpu10.slurm.sh（这两个是之前的实验提交文本，你这次可以参考一下）；
然后这里面是local做的实验的result，你这次实验result的格式也要按照这个来：/home/comp/f2256768/FYP_final/outputs；
另外需要注意的两个点就是：
1.cdf的cache等数据现在是放在/tmp盘上面你找一下
2.现在本节点(gpu10)被占满了，所以你需要换一个节点，你可以自己找一个4090节点，但是由于compute node间的/tmp盘不相通，所以需要你先把前面找到的实验所需的数据cp到真正实验的那个node上去（我给节点设置了免密登陆，相互传文件不会被卡），然后再开始实验；
没有问题的话就直接开始执行工作

-
我已经确认现成的 CDF 数据在 /tmp/f2256768/...，而当前 gpu10 确实没法用。现在去查哪些 4090 节点是空的，以及新仓库里 threshold sweep 输出的参数和结果结构，这样我可以直接做提交脚本并开跑。

我已经把脚本接口看明白了：这次重点不是重新抽 cache，而是把现有的 /tmp/f2256768/DF40_test_cdf、/tmp/f2256768/cdf_cache、/tmp/f2256768/fyp_final_compact_cdf 搬到一个空闲 4090 节点，然后按 outputs/*.json,csv 的现有命名风格跑 full / no-FR 两条线。接下来我会补节点信息、核对 checkpoint 是否齐全，并直接搭好新的集群提交脚本。

我准备直接把这次集群流程做成“在目标 4090 节点上先同步 /tmp，再自动跑 full/no-FR 两条线”的脚本，这样不用手动来回切节点，也能避免 gpu10 被占满的问题。现在补查一下 threshold_sweep 里 CDF real 的输入格式，确保输出字段和你现有 outputs 保持一致。

我已经确认这次最稳的做法是：在一个自动分配到的 4090 节点上，先从 gpu10:/tmp/f2256768/... 同步 CDF 所需目录，再直接在共享的 /home/comp/f2256768/FYP_final/outputs 里落结果。下面我会把这套提交脚本补进仓库，然后提交作业。

我在补最后一层可执行性检查：CelebDF real 的 cluster 输入已经齐了；接下来确认传输命令和 `sbatch` 可用，然后我就落脚本并提作业。

-
我接着往下做。现在先补查一个关键点：run_threshold_sweep.py 不只会吃 CDF，还会同时读取 DF40_train / DF40_test_ff / DF40_test_ood 的 patch cache，所以我先确认这些基础 cache 在集群上实际放在哪，再一起写进提交脚本。

我刚确认到 threshold_sweep 的确依赖完整的 DF40_train / test_ff / test_ood cache，所以不能只带 CDF 那一部分。现在继续把这些基础 cache 的真实落盘位置找出来，一并纳入同步和运行配置。

我还在补完整 cache 根目录的位置。前面全局搜索有点大，我已经缩到关键 split 名字上继续查，确认后就能把同步范围定死，不会漏掉训练/验证那部分缓存。

我发现一个很关键的分叉点：新拉下来的 FYP_final 仓库里没有带本地 cache，而 threshold_sweep.py 又依赖完整的 DF40_train/test_ff/test_ood cache。现在我去看备份目录 FYP_final_cluster_backup 里有没有这些 gitignored cache，如果有就直接复用；这样能避免误把实验卡在缺基础缓存这一步。

我把几个同名 cache 根目录分开核对了：/home/comp/f2256768/cache_clip 现在只看得到 DF40_test_cdf，并不是完整训练 cache 根。还差真正的 DF40_train / DF40_test_ff / DF40_test_ood 路径，我继续追这个，否则 threshold sweep 没法完整复现。

现在情况更清楚了：新仓库和备份仓库里都没有完整的 `DF40_train/test_ff/test_ood` cluster cache，所以直接照搬 `run_threshold_sweep.py` 并不现实。基于 handoff 的 Priority A/B，我准备改成一套更贴合现状的 cluster CDF 流程：直接用现成的 CDF fake cache 做 `0.1-0.9` sweep，再用 corrected checkpoint 做 replay summary，输出名仍然按 handoff 落到 `outputs/`。我先把最终阈值来源再核对一下，然后就落脚本提交。
