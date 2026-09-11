import argparse
import os

import torch

from glob import glob
from tqdm import tqdm

from simpy import Environment, Resource, Container, Store
try:
    from .arch import *
    from .kernel import *
except ImportError:  # Preserve direct execution from the original source tree.
    from arch import *
    from kernel import *


def dump_log(log_path: str, log: str):
    with open(log_path, "a") as f:
        f.write(log)


class Simulator:
    def __init__(self):
        self.env = Environment()
        self.simulation_log = []
        self.dram = Dram(self.env, 1, 1)
        self.scratchpad = Scratchpad(self.env, 32, 32)
        self.cores = [
            Core(self.env, self.scratchpad, self.dram, i) for i in range(N_CORE)
        ]
        self.dma = DMA(self.env, self.scratchpad, self.dram, self.cores)

        self.cycles = 0
        self.ops = 0
        self.dram_access = 0
        self.scratchpad_access = 0
        self.local_sram_access = 0

        self.cycles_diffq = 0
        self.cycles_mpmma = 0
        self.cycles_rdiff = 0
        self.ffn_throughput = []
        self.attention_throughput = []

    def linear(
        self,
        shape: tuple,
        nbit_map: torch.Tensor,
        diff: bool = False,
        gelu: bool = False,
    ):
        runtime_stat = {
            "ops": 0,
            "dram_access": 0,
            "scratchpad_access": 0,
            "local_sram_access": 0,
        }
        cycles_diffq = 0
        cycles_mpmma = 0
        cycles_rdiff = 0
        ffn_throughput = 0

        m, n, k = shape

        if diff:

            for index_core, core in enumerate(self.cores):
                self.env.process(
                    core.launch(
                        runtime_stat,
                        krnl_quantize(index_core, (m, k), nbit_map, diff, (64, 128)),
                    )
                )
            start_timestep = self.env.now
            self.env.run()
            end_timestep = self.env.now
            cycles_diffq += end_timestep - start_timestep  # diffq

        for index_core, core in enumerate(self.cores):
            self.env.process(
                core.launch(
                    runtime_stat,
                    krnl_quantize_gemm_wrap(index_core, shape, nbit_map, diff, gelu),
                )
            )
        self.env.process(self.dma.launch(runtime_stat, krnl_gemm_dma(shape, nbit_map)))

        start_timestep = self.env.now
        self.env.run()
        end_timestep = self.env.now
        
        
        if diff:
            cycles_mpmma += end_timestep - start_timestep
            ffn_throughput = nbit_map.reshape(-1).sum(dim=-1) / 2 * 64 * n / cycles_mpmma
            

        if diff:
            cycles_rdiff += int(cycles_diffq * k / m / 2)  # rdiff

        self.ops += runtime_stat["ops"] / 1e9
        self.dram_access += runtime_stat["dram_access"] / 1e9
        self.scratchpad_access += runtime_stat["scratchpad_access"] / 1e9
        self.local_sram_access += runtime_stat["local_sram_access"] / 1e9

        self.cycles_diffq += cycles_diffq
        self.cycles_mpmma += cycles_mpmma
        self.cycles_rdiff += cycles_rdiff
        
        if diff:
            self.ffn_throughput.append( ffn_throughput )

        return f" ================ Layer : Linear ================ \n \
            Shape : {shape}, \n \
            Diff : {diff}, \n \
            Gelu : {gelu}, \n \
            Ops : {runtime_stat["ops"]}, \n \
            Dram_access : {runtime_stat["dram_access"]}, \n \
            Scratchpad_access : {runtime_stat["scratchpad_access"]}, \n \
            Local_sram_access : {runtime_stat["local_sram_access"]}, \n \
            Cycles_diffq : {cycles_diffq}, \n \
            Cycles_mpmma : {cycles_mpmma}, \n \
            Cycles_rdiff : {cycles_rdiff}, \n \
            Throughput_ffn : {ffn_throughput}, \n \
            Cycles : {end_timestep-start_timestep} \n ================================================================ \n"

    def attention(
        self,
        shape: tuple,
        attn_mask: torch.Tensor,
        reuse: bool = False,
    ):
        b, h, n, k = shape
        runtime_stat = {
            "ops": 0,
            "dram_access": 0,
            "scratchpad_access": 0,
            "local_sram_access": 0,
        }
        for index_core, core in enumerate(self.cores):
            if reuse:
                self.env.process(
                    core.launch(
                        runtime_stat,
                        krnl_attentio_reuse_wrap(index_core, shape, attn_mask),
                    )
                )
            else:
                self.env.process(
                    core.launch(
                        runtime_stat, krnl_attention_wrap(index_core, shape, attn_mask)
                    )
                )
        if reuse:
            self.env.process(
                self.dma.launch(runtime_stat, krnl_attentio_reuse_dma(shape, attn_mask))
            )
        else:
            self.env.process(
                self.dma.launch(runtime_stat, krnl_attention_dma(shape, attn_mask))
            )

        start_timestep = self.env.now
        self.env.run()
        end_timestep = self.env.now
        
        attention_throughput = (attn_mask.reshape(-1).sum(dim=-1) * 64*64*k*2*4) / (end_timestep-start_timestep)

        self.ops += runtime_stat["ops"] / 1e9
        self.dram_access += runtime_stat["dram_access"] / 1e9
        self.scratchpad_access += runtime_stat["scratchpad_access"] / 1e9
        self.local_sram_access += runtime_stat["local_sram_access"] / 1e9
        self.attention_throughput.append(attention_throughput)

        return f" ================ Layer : Attention ================ \n \
            Shape : {shape}, \n \
            Reuse : {reuse}, \n \
            Ops : {runtime_stat["ops"]}, \n \
            Dram_access : {runtime_stat["dram_access"]}, \n \
            Scratchpad_access : {runtime_stat["scratchpad_access"]}, \n \
            Local_sram_access : {runtime_stat["local_sram_access"]}, \n \
            Throughput_attention : {attention_throughput}, \n \
            Cycles : {end_timestep-start_timestep} \n ================================================================ \n"


# def main():
#     # Setup simulator

#     env = Environment()

#     dram = Dram(env, 1, 1)
#     scratchpad = Scratchpad(env, 32, 32)
#     cores = [Core(env, scratchpad, dram, i) for i in range(N_CORE)]
#     dma = DMA(env, scratchpad, dram, cores)

#     # Launch tasks

#     shape = (4096, 6144, 1536)

#     env.process(dma.launch(krnl_dma_ctrl(shape)))
#     for id_core, core in enumerate(cores):
#         env.process(core.launch(krnl_linear_quantize_8x8(shape)))

#     env.run()

#     # for idx, inst in enumerate(krnl_linear_quantize_8x8(shape)):
#     #     print(idx, inst.opcode)

#     print(env.now)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", type=str)
    parser.add_argument("--log_path", type=str)
    args = parser.parse_args()

    trace_path = args.trace_path
    # trace_path = "/home/czhang/adapt/trace/sd3.5-medium/int8/"
    trace_files = glob(f"{trace_path}*.trace")

    simulator = Simulator()

    # simulator.attention((2, 16, 16, 64), torch.full((2, 24, 64, 64), True), False)
    # simulator.linear((8192, 1536, 6144), torch.full((128, 48), 128 * 8), False, False)
    # simulator.linear((8192, 1536, 1536), torch.full((128, 24), 8), False, False)
    # simulator.linear((8192, 1536, 1536), torch.full((128, 24), 8), False, False)
    # simulator.linear((8192, 1536, 1536), torch.full((128, 24), 8), False, False)

    # print(simulator.env.now)

    # for idx, inst in enumerate(
    #     krnl_diff_quantize(0, (4096, 1536), torch.full((64, 1536), 8))
    # ):
    #     print(idx, inst.opcode)

    for i, t in enumerate(trace_files):
        print(f"{i} / {len(trace_files)} : ", t)
        trace = torch.load(t)
        match trace["layer"]:
            case "attention":
                log = simulator.attention(
                    trace["shape"],
                    trace["attn_mask"],
                    trace["reuse"],
                )
                dump_log(args.log_path, log)
                # print(trace["attn_mask"].shape,trace["attn_mask"].sum())
            case "linear":
                log = simulator.linear(
                    trace["shape"],
                    trace["nbit_map"],
                    trace["diff"],
                    trace["gelu"],
                )
                dump_log(args.log_path, log)
                pass
            case _:
                raise NotImplementedError("!")

    dump_log(
        args.log_path,
        f"{simulator.env.now}, {simulator.ops}, {simulator.dram_access}, {simulator.scratchpad_access}, \
        {simulator.local_sram_access},{simulator.cycles_diffq},{simulator.cycles_mpmma}, {simulator.cycles_rdiff}, \
        {sum(simulator.ffn_throughput)/len(simulator.ffn_throughput)},\
        {sum(simulator.attention_throughput)/len(simulator.attention_throughput)}",
    )

    print(
        simulator.env.now,
        simulator.ops,
        simulator.dram_access,
        simulator.scratchpad_access,
        simulator.local_sram_access,
        simulator.cycles_diffq,
        simulator.cycles_mpmma,
        simulator.cycles_rdiff,
    )
