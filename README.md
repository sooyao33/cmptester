# CMP 170HX 解锁 + 稳定性测试

使用 [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)（最新版本）解锁 NVIDIA CMP 170HX 显卡，
然后通过实际测量和日志证明解锁成功、显卡运行稳定。

**要求：** Ubuntu/Debian 系统，x86-64，需要 root 权限（sudo），需要联网，**BIOS 中的安全启动（Secure Boot）必须关闭**。

(English version: [README.en.md](README.en.md))

## 使用方法（卖家）

**第 1 步：** 打开终端，复制粘贴以下命令，按回车：

```bash
git clone https://github.com/sooyao33/cmptester.git && cd cmptester && sudo ./run.sh 1
```

（`1` = 测试 1 小时。需要输入密码时，输入您的登录密码。）

**第 2 步：** 等待约 10–20 分钟。**电脑会自动关机。**

**第 3 步：** **按电源键开机。**
（如果 3 分钟后还没有关机：长按电源键 10 秒，再按一次开机。）

**第 4 步：** 什么都不用做。开机后测试自动进行，约 1 小时 15 分钟完成。

**第 5 步：** 把这个文件发给我们：

```
/root/cmp-test-results-<主机名>.tar.gz
```

（您的用户主目录 `~` 下也有一份同样的文件。）

### 查看进度（可选）

```bash
sudo /opt/cmp-test/run.sh status
```

- 测试在后台运行，关闭终端窗口不会中断测试。
- 如果电脑自动重启了一次（不是关机），这是正常的，测试会自动继续。
- 如果屏幕显示 "STOPPED WITH AN ERROR"：请把 `/var/log/cmp-test.log` 文件发给我们。

---

## 技术细节

所有检测项目都已在两张已知良好的解锁卡上验证过（2026-10-31），包括故意注入错误：
检测器必须先能抓到人为植入的错误，"无错误"的结果才有意义。

程序已为 sm_80 预编译（CUDA 13.3，静态运行库；需要 610.x 驱动，第 1 阶段会自动安装）。源代码在 `src/` 目录。

| 阶段 | 命令 | 是否修改系统 | 时间 |
|---|---|---|---|
| 0 信息收集 | `sudo ./00-recon.sh` | 否（只读） | 10 秒 |
| 1 解锁 | `sudo ./01-unlock.sh` | 是：编译工具、nvidia-open 610.43.03（或 cmpunlocker 支持的最新版本）、最新 cmpunlocker（`--no-passthrough --no-iommu`） | 10–20 分钟 |
| -- | **冷启动**（关机后由卖家按电源键开机） | | |
| 2 验证 | `sudo ./02-verify-unlock.sh` | 否 | 约 5 分钟 |
| 3 自检 | `sudo ./03-selftest.sh` | 否 | 约 5 分钟 |
| 4 稳定性测试 | `sudo ./04-soak.sh start 小时数` / `status` / `stop` | 否 | 所选小时数 |
| 5 证据汇总 | `sudo ./05-collect.sh` → `out/PROOF.md`（含每张卡的显存带宽、FP32/FP16 TFLOPS、功耗、温度、频率、降频原因）+ 带 SHA256SUMS 的压缩包 | 否 | 1 分钟 |

`run.sh` 会自动按顺序执行以上所有阶段。

## 证明什么，如何证明

| 要证明的内容 | 证据 | 通过标准（参考卡） |
|---|---|---|
| 运行的是解锁版驱动 | `/sys/module/nvidia/srcversion` 与编译的模块一致；SEC2_DEBUG 启动日志；无 RmInitAdapter 失败 | 完全一致 |
| 显存容量真实存在，而不只是显示数字 | cmpprobe 分配所有 1 GiB 显存块，写入由地址生成的数据，用非缓存读取校验，2 轮（第二轮数据取反） | ≥ 60 GiB，0 个错误（参考：61 GiB） |
| 检测程序能发现错误 | `--inject` 在 8 GiB 以下和以上各植入一个位翻转 | 恰好抓到 2 个 |
| 所有 SM 已解锁 | CUDA SM 数量，FP32 FMA 吞吐量 | 74 个 SM，≥ 12.0 TFLOPS（参考 13.1） |
| HBM 显存全速 | 拷贝内核带宽 | ≥ 1100 GB/s（参考 1195–1221） |
| Tensor Core 计算正确 | 用精确整数做 FP16 WMMA 矩阵乘，与独立的 CUDA Core 矩阵乘逐元素比较（1670 万个元素 × 5 次）+ CPU 抽查 | 逐位一致，≥ 10 TFLOPS fp16（此简单内核参考值 12.3） |
| 检测器在这台机器上有效 | gpuqual `--self-test`（每轮植入：软错误、坏单元、计算错误），vramcheck `--inject` | 每个植入的错误都被正确分类 |
| 长时间稳定 | gpuqual 测试框架：黄金值重算校验、移动反转、读改写压力、随机散列读取、ALU 已知答案测试、高温下数据保持测试；监控 Xid 和 PCIe AER；每 30 秒记录健康数据（含 ECC 计数） | 报告结论不是 NO-GO；0 个数据错误；0 个 Xid；ECC 计数不变 |

**关于 ECC：** 新版解锁会开启 ECC。ECC 会自动纠正单个位翻转，所以有问题的卡在数据测试中也可能显示正常。
因此采样程序会记录 ECC 计数，`05-collect.sh` 会对比测试前后：即使 gpuqual 没有发现错误，ECC 纠错计数上升也算一个问题。

**测试时长：** gpuqual 完整一轮在正常速度下约为 3.25 小时。较短的测试会按比例压缩为正好一轮，所有阶段（包括数据保持测试）仍会执行；
测试时间越长，证据越有力。温度保护：GPU 85 °C / HBM 92 °C 时停止测试；测试程序从不修改功耗限制（只记录当前值）。

**安全性：** 安全启动开启时第 1 阶段会拒绝运行；不会修改 GRUB/内核启动参数（`--no-iommu`），不设置虚拟机直通（`--no-passthrough`）。
已安装的不受支持的 NVIDIA 驱动会被自动卸载并替换。安装解锁时会让原驱动保持占用，避免热替换模块；之后的冷启动让显卡以解锁驱动启动。
驱动加载失败后，不要热重启，要完全断电重启（热重启曾导致我们的参考机器死机）。

**恢复原状（卖家机器）：** `cd /opt/cmpunlocker && sudo ./remove.sh --yes`，
如需同时删除驱动再执行 `sudo nvidia-uninstall`，然后重启。
