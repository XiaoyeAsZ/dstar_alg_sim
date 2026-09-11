from collections import deque
from typing import Iterable
from simpy.events import AllOf, AnyOf, Event
from simpy import Environment, Resource, Container, Store
from simpy.resources.resource import Request


ENABLE_PRINT_TRACE = True

# ---------- TPU Component Parameters ----------
ACTIVATION_BUF_SIZE = 4096
WEIGHT_BUF_SIZE = 4096
OUTPUT_BUF_SIZE = 4096

SRAM_LATENCY = 1  # cycles
DRAM_LATENCY = 10  # cycles
MXU_LATENCY = 5  # cycles per compute task
NUM_MXU_UNITS = 1  # systolic array capacity

CLOCK_PERIOD = 1  # 1 ns
DRAM_BANDWIDTH = 1500  # 1500 GB/s
SCRATCHPAD_BANDWIDTH = 256  # 256 GB/s Per bank
L2_SIZE = 512 * 1024  # 512 KB
N_CORE = 32

# ---------- TPU Process Definitions ----------


def HEX2INT(addr: str) -> int:
    return int(addr, 16)


# Global Address Mapping
# 40GB DRAM
DRAM_ADDR_START = HEX2INT("0x0000000000000000")
# 512KB Scratchpad
SCRATCHPAD_ADDR_START = HEX2INT("0x0000000A00000000")
UNMAPPED_ADDR_START = HEX2INT("0x0000000A20000000")

# Local Address Mapping
# UNIBUF0_ADDR_START = HEX2INT("0x0000")
# UNIBUF1_ADDR_START = HEX2INT("0x2000")
# UNIBUF2_ADDR_START = HEX2INT("0x4000")
# UNIBUF3_ADDR_START = HEX2INT("0x6000")
# UNIBUF4_ADDR_START = HEX2INT("0x8000")
# UNIBUF5_ADDR_START = HEX2INT("0xA000")
# UNIBUF6_ADDR_START = HEX2INT("0xC000")
# UNIBUF7_ADDR_START = HEX2INT("0xE000")
# UNIBUF8_ADDR_START = HEX2INT("0x10000")
# UNIBUF9_ADDR_START = HEX2INT("0x12000")
# UNIBUF10_ADDR_START = HEX2INT("0x14000")

UNIBUF0_ADDR_START = 0
UNIBUF1_ADDR_START = 8192 * 1
UNIBUF2_ADDR_START = 8192 * 2
UNIBUF3_ADDR_START = 8192 * 3
UNIBUF4_ADDR_START = 8192 * 4
UNIBUF5_ADDR_START = 8192 * 5
UNIBUF6_ADDR_START = 8192 * 6
UNIBUF7_ADDR_START = 8192 * 7
UNIBUF8_ADDR_START = 8192 * 8
UNIBUF9_ADDR_START = 8192 * 9
UNIBUF10_ADDR_START = 8192 * 10
UNIBUF11_ADDR_START = 8192 * 11
UNIBUF12_ADDR_START = 8192 * 12
UNIBUF13_ADDR_START = 8192 * 13
UNIBUF14_ADDR_START = 8192 * 14
UNIBUF15_ADDR_START = 8192 * 15
UNIBUF16_ADDR_START = 8192 * 16
UNIBUF17_ADDR_START = 8192 * 17
UNIBUF18_ADDR_START = 8192 * 18
UNIBUF19_ADDR_START = 8192 * 19
UNIBUF20_ADDR_START = 8192 * 20
UNIBUF21_ADDR_START = 8192 * 21


class ScoreboardItem:
    def __init__(self, event: Event, opcode: str, src1: int, src2: int, dest: int):
        self.event = event
        self.opcode = opcode
        self.src1 = src1
        self.src2 = src2
        self.dest = dest


class Dram:
    def __init__(
        self,
        env: Environment,
        # capacity: int,
        nRP: int = 1,
        nWP: int = 1,
    ):
        # self.buffer = Store(env, capacity)
        self.readPort = Container(env, nRP, nRP)
        self.writePort = Container(env, nWP, nWP)


class Scratchpad:
    def __init__(
        self,
        env: Environment,
        # capacity: int,
        nRP: int = 1,
        nWP: int = 1,
    ):
        # self.buffer = Store(env, capacity)
        self.readPort = Container(env, nRP, nRP)
        self.writePort = Container(env, nWP, nWP)


class Sram:
    def __init__(
        self,
        env: Environment,
        nRP: int,
        nWP: int,
        capacity: int = 8192,
        segment: int = 4096,
    ):
        assert capacity % segment == 0
        self.buffer = [Container(env, 1, 0) for _ in range(capacity // segment)]
        # self.readPort = Container(env, nRP)
        # self.writePort = Container(env, nWP)


class UniBuf:
    def __init__(self, env: Environment, n_bank: int, capacity: int):
        self.n_bank = n_bank
        self.capacity = capacity
        self.sram_array = [Sram(env, 1, 1, capacity, 4096) for i in range(n_bank)]

    def addr_map(self, base_addr: int) -> list[int, int]:
        assert base_addr < self.n_bank * self.capacity
        assert base_addr % 4096 == 0
        index_bank = base_addr // self.capacity
        index_segment = (base_addr % self.capacity) // 4096
        return index_bank, index_segment

    def request(self, base_addr: int) -> Container:
        index_bank, index_segment = self.addr_map(base_addr)
        # return (
        #     self.sram_array[index_bank].readPort,
        #     self.sram_array[index_bank].writePort,
        #     self.sram_array[index_bank].buffer[index_segment],
        # )
        return self.sram_array[index_bank].buffer[index_segment]


class Core:
    def __init__(
        self,
        env: Environment,
        scratchpad: Sram,
        dram: Dram,
        id_core: int,
        inst_queue_capacity: int = 4,
    ):
        self.id_core = id_core

        self.env = env
        self.scratchpad = scratchpad
        self.dram = dram

        self.inst_queue_capacity = inst_queue_capacity
        self.scoreboard = deque(maxlen=inst_queue_capacity)

        self.load_event = deque()

        # Mem interface
        self.mem_read_queue = Container(env, 16, 16)
        self.mem_read_ctrl = Container(env, 1, 1)
        self.mem_write_queue = Container(env, 16, 16)
        self.mem_write_ctrl = Container(env, 1, 1)

        # Systolic array
        self.mxu_queue = Container(env, 16, 16)
        self.mxu_sa = Container(env, 1, 1)
        self.mxu_invq = Container(env, 1, 1)
        self.mxu_rec = Container(env, 1, 1)

        self.inverse_quant_unit = Container(env, 1, 1)

        # Quantization process Unit
        self.qpu_queue = Container(env, 16, 16)
        self.qpu_diff = Container(env, 1, 1)
        self.qpu_rec = Container(env, 1, 1)
        self.qpu_quantize = Container(env, 1, 1)
        self.qpu_sort = Container(env, 1, 1)

        # VPU
        # self.element_wise_unit = Container(env, 1, 0)
        # self.vector_vector_unit = Container(env, 1, 0)
        # self.vector_scalar_unit = Container(env, 1, 0)
        # self.reduction_unit = Container(env, 1, 0)
        # self.sorting_unit = Container(env, 1, 0)
        self.vpu_queue = Container(env, 16, 16)
        self.vpu_ewise = Container(env, 1, 1)
        self.vpu_rec = Container(env, 1, 1)
        self.vpu_vs = Container(env, 1, 1)

        # self.actBuf = Sram(env, 1, 1)
        # self.weightBuf = Sram(env, 1, 1)
        # self.outputBuf = Sram(env, 1, 1)
        # self.uniBuf = Sram(env, 4, 4)
        self.uni_buf = UniBuf(env, 22, 8192)

        self.interrupt = Container(env, 1, 0)

    def launch(self, runtime_stat: dict, kernel: Iterable):
        for index, inst in enumerate(kernel):
            # Check whether inst queue can hold new inst
            if len(self.scoreboard) == self.inst_queue_capacity:
                yield AnyOf(self.env, [s.event for s in self.scoreboard])

            # Check inst dependency
            # if inst.src1 is not None:
            #     inst_dependency = []
            #     for item in self.scoreboard:
            #         if item.dest == inst.src1 or item.dest == inst.src2:
            #             inst_dependency.append(item.event)
            #     yield AllOf(self.env, inst_dependency)

            # if self.id_core == 0:
            #     print(f"[{self.env.now}] Issuing inst : {inst.opcode}")

            match inst.opcode:
                case "SYNC":
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    yield AllOf(self.env, self.load_event)
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    yield self.interrupt.get(1)

                # case "FENCE":
                #     yield AllOf(self.env, [s.event for s in self.scoreboard])
                case "LOAD":
                    yield self.mem_read_queue.get(1)
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    self.env.process(
                        inst.execute(
                            index,
                            runtime_stat,
                            self.env,
                            self,
                            self.scratchpad,
                            self.dram,
                        )
                    )

                case "STORE":
                    yield self.mem_write_queue.get(1)
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    self.env.process(
                        inst.execute(
                            index,
                            runtime_stat,
                            self.env,
                            self,
                            self.scratchpad,
                            self.dram,
                        )
                    )
                case "QUANTIZE":
                    yield self.qpu_queue.get(1)
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    self.env.process(inst.execute(index, runtime_stat, self.env, self))
                case "DIFFQUANTIZE":
                    yield self.qpu_queue.get(1)
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    self.env.process(inst.execute(index, runtime_stat, self.env, self))
                case "GELU":
                    yield self.vpu_queue.get(1)
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    self.env.process(inst.execute(index, runtime_stat, self.env, self))
                case "SOFTMAX":
                    yield self.vpu_queue.get(1)
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    self.env.process(inst.execute(index, runtime_stat, self.env, self))
                case "MMA":
                    yield self.mxu_queue.get(1)
                    # if self.id_core == 0:
                    #     print(f"[{self.env.now}] Issuing inst {index} : {inst.opcode}")
                    self.env.process(inst.execute(index, runtime_stat, self.env, self))
                case _:
                    raise NotImplementedError("Unsupported instruction.")
            yield self.env.timeout(1)


class DMA:
    def __init__(
        self,
        env: Environment,
        scratchpad: Sram,
        dram: Dram,
        cores: list[Core],
    ):
        self.env = env
        self.scratchpad = scratchpad
        self.dram = dram
        self.cores = cores

        self.mem_ctrl = Resource(env, 1)

    def launch(self, runtime_stat: dict, kernel: Iterable):
        for index, inst in enumerate(kernel):
            # print(self.env.now, index, inst.opcode)
            yield self.env.process(
                inst.execute(
                    self.env, runtime_stat, self, self.scratchpad, self.dram, self.cores
                )
            )
