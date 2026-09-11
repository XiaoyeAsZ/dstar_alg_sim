from simpy import Environment, Resource, Container
try:
    from .arch import *
except ImportError:  # Preserve direct execution from the original source tree.
    from arch import *


class Instruction:
    def __init__(self):
        self.opcode = None
        self.addr = None
        self.src1 = None
        self.src2 = None
        self.dest = None

    def execute(self):
        raise NotImplementedError("!")


class SYNC(Instruction):
    def __init__(self):
        super().__init__()
        self.opcode = "SYNC"

    def execute(self, env: Environment, core: Core, inst_queue: Store):
        pass


class LOAD(Instruction):
    def __init__(
        self,
        addr: int,
        dest: int,
        data_size: int,
    ):
        super().__init__()
        self.opcode = "LOAD"
        self.addr = addr
        self.dest = dest
        self.data_size = data_size

    def execute(
        self,
        id_inst: int,
        runtime_stat: dict,
        env: Environment,
        core: Core,
        scratchpad: Scratchpad = None,
        dram: Dram = None,
    ):
        # assert not ((scratchpad is not None) and (dram is not None))

        event = Event(env)
        core.load_event.append(event)

        if self.addr < SCRATCHPAD_ADDR_START:
            mem_src = dram
            bandwidth = DRAM_BANDWIDTH
            runtime_stat["dram_access"] += self.data_size
            # if core.id_core == 0:
            #     print("LOAD read DDR ", self.data_size)
        elif self.addr >= SCRATCHPAD_ADDR_START and self.addr < UNMAPPED_ADDR_START:
            mem_src = scratchpad
            bandwidth = SCRATCHPAD_BANDWIDTH
            runtime_stat["scratchpad_access"] += self.data_size
        else:
            raise NotImplementedError("!")

        # sb_item = ScoreboardItem(env.event(), "LOAD", self.src1, self.src2, self.dest)
        # core.scoreboard.append(sb_item)
        # inst_dependency = []
        # for item in core.scoreboard:
        #     if (item is not sb_item) and (
        #         self.dest == item.src1 or self.dest == item.src2
        #     ):
        #         inst_dependency.append(item.event)
        # yield AllOf(env, inst_dependency)

        yield core.mem_read_ctrl.get(1)
        yield mem_src.readPort.get(1)

        yield env.timeout(self.data_size // bandwidth)

        yield mem_src.readPort.put(1)

        yield core.uni_buf.request(self.dest).put(1)

        yield core.mem_read_ctrl.put(1)

        yield core.mem_read_queue.put(1)

        event.succeed()
        core.load_event.remove(event)

        # yield core.uni_buf.request(self.dest).put(1)

        # sb_item.event.succeed()
        # core.scoreboard.remove(sb_item)

        # print("LOAD finish")


class STORE(Instruction):
    def __init__(
        self,
        src1: int,
        addr: int,
        data_size: int,
    ):
        super().__init__()
        self.opcode = "STORE"
        self.addr = addr
        self.src1 = src1
        self.data_size = data_size

    def execute(
        self,
        id_inst: int,
        runtime_stat: dict,
        env: Environment,
        core: Core,
        scratchpad: Scratchpad = None,
        dram: Dram = None,
    ):
        if self.addr < SCRATCHPAD_ADDR_START:
            mem_dest = dram
            bandwidth = DRAM_BANDWIDTH
            runtime_stat["dram_access"] += self.data_size
            # if core.id_core == 0:
            #     print("STORE read DDR ", self.data_size)
        elif self.addr >= SCRATCHPAD_ADDR_START and self.addr < UNMAPPED_ADDR_START:
            mem_dest = scratchpad
            bandwidth = SCRATCHPAD_BANDWIDTH
            runtime_stat["scratchpad_access"] += self.data_size
        else:
            raise NotImplementedError("!")

        # sb_item = ScoreboardItem(env.event(), "STORE", self.src1, self.src2, self.dest)
        # core.scoreboard.append(sb_item)
        # inst_dependency = []
        # for item in core.scoreboard:
        #     if (item is not sb_item) and (item.dest == self.src1):
        #         inst_dependency.append(item.event)
        # yield AllOf(env, inst_dependency)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "Store issued ", self.src1)

        yield core.mem_write_ctrl.get(1)
        yield mem_dest.writePort.get(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "Store got port ", self.src1)

        yield core.uni_buf.request(self.src1).get(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "Store start ", self.src1)

        yield env.timeout(self.data_size // bandwidth)

        yield mem_dest.writePort.put(1)

        yield core.mem_write_ctrl.put(1)
        yield core.mem_write_queue.put(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "Store ", self.src1)

        # sb_item.event.succeed()
        # core.scoreboard.remove(sb_item)


class MMA(Instruction):
    def __init__(
        self,
        src1: int,
        src2: int,
        dest: int,
        k: int,
        latency: int,
        last: bool = False,
    ):
        super().__init__()
        self.opcode = "MMA"
        self.src1 = src1
        self.src2 = src2
        self.dest = dest
        self.k = k
        self.latency = latency
        self.last = last

    def execute(
        self,
        id_inst: int,
        runtime_stat: dict,
        env: Environment,
        core: Core,
    ):

        runtime_stat["local_sram_access"] += 64 * self.k
        runtime_stat["local_sram_access"] += 64 * self.latency
        runtime_stat["local_sram_access"] += 64 * self.k * 2

        runtime_stat["ops"] += 64 * self.latency * 64

        # Stage 1 : Matmul

        # if core.id_core == 0:
        #     print(env.now, "MMA stage1")

        yield core.mxu_sa.get(1)

        yield core.uni_buf.request(self.src1).get(1)

        yield core.uni_buf.request(self.src2).get(1)

        yield env.timeout(self.latency)
        yield core.mxu_invq.get(1)
        yield core.mxu_sa.put(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "MMA stage 1 finish")

        # Stage 2 : Inv Quantization
        yield env.timeout(64)
        yield core.mxu_rec.get(1)
        yield core.mxu_invq.put(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "MMA stage 2 finish")

        # Stage 3 : Accumulation
        yield env.timeout(64)

        if self.last:
            yield core.uni_buf.request(self.dest).put(1)

        yield core.mxu_rec.put(1)

        yield core.mxu_queue.put(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "MMA finish after ", self.latency + 64 + 64)


class QUANTIZE(Instruction):
    def __init__(self, src1: int, dest: int, k: int = 64):
        super().__init__()
        self.opcode = "QUANTIZE"
        self.src1 = src1
        self.dest = dest
        self.k = k

    def execute(
        self,
        id_inst: int,
        runtime_stat: dict,
        env: Environment,
        core: Core,
    ):
        runtime_stat["local_sram_access"] += 64 * self.k * 2
        runtime_stat["local_sram_access"] += 64 * self.k * 1

        yield core.qpu_rec.get(1)
        yield core.uni_buf.request(self.src1).get(1)

        # Stage 2 : Find absmax
        yield env.timeout(self.k + 6)
        yield core.qpu_quantize.get(1)
        yield core.qpu_rec.put(1)

        # Stage 3 : Quantize
        yield env.timeout(self.k)
        yield core.qpu_sort.get(1)
        yield core.qpu_quantize.put(1)

        # Stage 4 : Sort meta
        yield env.timeout(36)
        yield core.uni_buf.request(self.dest).put(1)
        yield core.qpu_sort.put(1)

        yield core.qpu_queue.put(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "QUANTIZE finish", self.src1, self.dest)


class DIFFQUANTIZE(Instruction):
    def __init__(self, src1: int, src2: int, dest: int, k: int = 64):
        super().__init__()
        self.opcode = "DIFFQUANTIZE"
        self.src1 = src1
        self.src2 = src2
        self.dest = dest
        self.k = k

    def execute(
        self,
        id_inst: int,
        runtime_stat: dict,
        env: Environment,
        core: Core,
    ):

        runtime_stat["local_sram_access"] += 64 * self.k * 2
        runtime_stat["local_sram_access"] += 64 * self.k * 2
        runtime_stat["local_sram_access"] += 64 * self.k * 1

        # Stage 1 : Diff
        yield core.qpu_diff.get(1)
        yield core.uni_buf.request(self.src1).get(1)
        yield core.uni_buf.request(self.src2).get(1)

        yield env.timeout(self.k)
        yield core.qpu_rec.get(1)
        yield core.qpu_diff.put(1)

        # Stage 2 : Find absmax
        yield env.timeout(self.k + 6)
        yield core.qpu_quantize.get(1)
        yield core.qpu_rec.put(1)

        # Stage 3 : Quantize
        yield env.timeout(self.k)
        yield core.qpu_sort.get(1)
        yield core.qpu_quantize.put(1)

        # Stage 4 : Sort meta
        yield env.timeout(36)
        yield core.uni_buf.request(self.dest).put(1)
        yield core.qpu_sort.put(1)

        yield core.qpu_queue.put(1)


class SOFTMAX(Instruction):
    def __init__(self, src1: int, dest: int):
        super().__init__()
        self.opcode = "SOFTMAX"
        self.src1 = src1
        self.dest = dest

    def execute(
        self,
        id_inst: int,
        runtime_stat: dict,
        env: Environment,
        core: Core,
    ):

        runtime_stat["local_sram_access"] += 64 * 64 * 2
        runtime_stat["local_sram_access"] += 64 * 64 * 2

        # if core.id_core == 0:
        #     print(env.now, id_inst, "SOFTMAX start", self.src1, self.dest)

        yield core.vpu_ewise.get(1)
        yield core.uni_buf.request(self.src1).get(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "SOFTMAX got resource", self.src1, self.dest)

        # Stage 1 : Exponent
        yield env.timeout(64)
        yield core.vpu_rec.get(1)
        yield core.vpu_ewise.put(1)

        # Stage 2 : Reduction
        yield env.timeout(6)
        yield core.vpu_vs.get(1)
        yield core.vpu_rec.put(1)

        # Stage 3 : Divide
        yield env.timeout(64)

        yield core.uni_buf.request(self.dest).put(1)
        yield core.vpu_vs.put(1)

        yield core.vpu_queue.put(1)

        # if core.id_core == 0:
        #     print(env.now, id_inst, "SOFTMAX finish", self.src1, self.dest)


class GELU(Instruction):
    def __init__(self, src1: int, dest: int, k: int = 64):
        super().__init__()
        self.opcode = "GELU"
        self.src1 = src1
        self.dest = dest
        self.k = k

    def execute(
        self,
        id_inst: int,
        runtime_stat: dict,
        env: Environment,
        core: Core,
    ):
        runtime_stat["local_sram_access"] += self.k * 64 * 2
        runtime_stat["local_sram_access"] += self.k * 64 * 2

        yield core.vpu_ewise.get(1)
        yield core.uni_buf.request(self.src1).get(1)

        yield env.timeout(self.k)

        yield core.uni_buf.request(self.dest).put(1)
        yield core.vpu_ewise.put(1)
        yield core.vpu_queue.put(1)


# DMA Instruction
class LOAD_DMA(Instruction):
    def __init__(self, data_size: int):
        super().__init__()
        self.opcode = "LOAD_DMA"
        self.data_size = data_size

    def execute(
        self,
        env: Environment,
        runtime_stat: dict,
        dma: DMA,
        scratchpad: Scratchpad,
        dram: Dram,
        cores: list[Core],
    ):
        runtime_stat["dram_access"] += self.data_size
        # print("DMA read DDR ", self.data_size)
        runtime_stat["scratchpad_access"] += self.data_size

        yield dram.readPort.get(1)
        yield scratchpad.writePort.get(1)

        yield env.timeout(self.data_size // DRAM_BANDWIDTH)
        # for core in cores:
        #     yield core.interrupt.put(1)

        yield dram.readPort.put(1)
        yield scratchpad.writePort.put(1)

        yield AllOf(env, [core.interrupt.put(1) for core in cores])
