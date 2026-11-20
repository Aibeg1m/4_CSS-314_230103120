import os
import sys
import json
import math
import hashlib
import subprocess
import numpy as np

from mpi4py import MPI
from numba import cuda, float64

TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
LAMBDA_DAMP = 0.15


def get_gpu_uuid():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8"
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user, gpu_uuid):
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def init_distributed_context(github_user):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(
            f"Expected exactly 4 MPI ranks, got {size}"
        )

    cuda.select_device(0)

    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)

    instance_seed = derive_student_seed(
        github_user, gpu_uuid
    )

    cart_comm = comm.Create_cart(
        dims=[2, 2],
        periods=[False, False],
        reorder=False
    )

    row_rank, col_rank = cart_comm.Get_coords(rank)

    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    print(
        f"Rank {rank}: coords=({row_rank},{col_rank}), "
        f"neighbors=({nbr_up},{nbr_down},"
        f"{nbr_left},{nbr_right})",
        flush=True
    )

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": rank,
        "size": size,
        "coords": (row_rank, col_rank),
        "neighbors": (
            nbr_up, nbr_down,
            nbr_left, nbr_right
        ),
        "gpu_uuid": gpu_uuid,
        "instance_seed": instance_seed
    }

def initialize_subdomains(context):
    row_rank, col_rank = context["coords"]
    seed = context["instance_seed"]

    nx = GLOBAL_NX // 2
    ny = GLOBAL_NY // 2

    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    rows = np.arange(ny + 2, dtype=np.float64)
    cols = np.arange(nx + 2, dtype=np.float64)

    y = (row_rank * ny + rows) * dy
    x = (col_rank * nx + cols) * dx

    X, Y = np.meshgrid(x, y)

    phase = (seed % 10000) / 10000.0

    kappa = (
        0.16
        + 0.04 * np.sin(2 * np.pi * X + phase)
        * np.cos(2 * np.pi * Y)
    )

    center_x = 0.3 + 0.2 * phase
    center_y = 0.4 + 0.2 * phase

    u = np.exp(
        -(
            (X - center_x) ** 2
            + (Y - center_y) ** 2
        ) / 0.01
    )

    source = 0.05 * np.exp(
        -(
            (X - 0.5) ** 2
            + (Y - 0.5) ** 2
        ) / 0.02
    )

    up, down, left, right = context["neighbors"]

    if up == MPI.PROC_NULL:
        u[0, :] = 0.0

    if down == MPI.PROC_NULL:
        u[-1, :] = 0.0

    if left == MPI.PROC_NULL:
        u[:, 0] = 0.0

    if right == MPI.PROC_NULL:
        u[:, -1] = 0.0

    d_u = cuda.to_device(u)
    d_kappa = cuda.to_device(kappa)
    d_source = cuda.to_device(source)

    print(
        f"Rank {context['rank']}: "
        f"local shape={d_u.shape}, "
        f"kappa max={np.max(kappa):.6f}",
        flush=True
    )

    return d_u, d_kappa, d_source

    # ============================================================
# TASK 3: CUDA HALO PACK / UNPACK
# ============================================================

# Directions:
# 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT

@cuda.jit
def pack_halos_kernel(u, buf, direction, ny, nx):
    idx = cuda.grid(1)

    if direction == 0:
        if idx < nx:
            buf[idx] = u[1, idx + 1]

    elif direction == 1:
        if idx < nx:
            buf[idx] = u[ny, idx + 1]

    elif direction == 2:
        if idx < ny:
            buf[idx] = u[idx + 1, 1]

    else:
        if idx < ny:
            buf[idx] = u[idx + 1, nx]


@cuda.jit
def unpack_halos_kernel(u, buf, direction, ny, nx):
    idx = cuda.grid(1)

    if direction == 0:
        if idx < nx:
            u[0, idx + 1] = buf[idx]

    elif direction == 1:
        if idx < nx:
            u[ny + 1, idx + 1] = buf[idx]

    elif direction == 2:
        if idx < ny:
            u[idx + 1, 0] = buf[idx]

    else:
        if idx < ny:
            u[idx + 1, nx + 1] = buf[idx]


def exchange_halos_async(context, d_u, stream_boundary):
    cart = context["cart_comm"]
    neighbors = context["neighbors"]

    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    directions = {
        0: (neighbors[0], nx),
        1: (neighbors[1], nx),
        2: (neighbors[2], ny),
        3: (neighbors[3], ny),
    }

    send_dev = {}
    recv_dev = {}
    send_host = {}
    recv_host = {}

    requests = []

    # Pack GPU boundaries and start async D2H copies.
    for direction, (neighbor, count) in directions.items():
        if neighbor == MPI.PROC_NULL:
            continue

        send_dev[direction] = cuda.device_array(count, dtype=np.float64)
        recv_dev[direction] = cuda.device_array(count, dtype=np.float64)

        send_host[direction] = cuda.pinned_array(count, dtype=np.float64)
        recv_host[direction] = cuda.pinned_array(count, dtype=np.float64)

        blocks = (count + 255) // 256

        pack_halos_kernel[blocks, 256, stream_boundary](
            d_u, send_dev[direction], direction, ny, nx
        )

        send_dev[direction].copy_to_host(
            send_host[direction],
            stream=stream_boundary
        )

    stream_boundary.synchronize()

    # Post receives first, then sends.
    for direction, (neighbor, count) in directions.items():
        if neighbor == MPI.PROC_NULL:
            continue

        opposite = direction ^ 1

        requests.append(
            cart.Irecv(
                recv_host[direction],
                source=neighbor,
                tag=opposite
            )
        )

    for direction, (neighbor, count) in directions.items():
        if neighbor == MPI.PROC_NULL:
            continue

        requests.append(
            cart.Isend(
                send_host[direction],
                dest=neighbor,
                tag=direction
            )
        )

    MPI.Request.Waitall(requests)

    # Copy received halos back to GPU and unpack.
    for direction, (neighbor, count) in directions.items():
        if neighbor == MPI.PROC_NULL:
            continue

        recv_dev[direction].copy_to_device(
            recv_host[direction],
            stream=stream_boundary
        )

        blocks = (count + 255) // 256

        unpack_halos_kernel[blocks, 256, stream_boundary](
            d_u, recv_dev[direction], direction, ny, nx
        )

    stream_boundary.synchronize()

# ============================================================
# TASK 4: SHARED-MEMORY TILED STENCIL KERNEL
# ============================================================

@cuda.jit
def tiled_stencil_kernel(
    u, u_new, kappa, source,
    dx, dy, dt,
    r_start, r_end, c_start, c_end
):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Position of this thread in the local subdomain
    row = r_start + cuda.blockIdx.y * 16 + ty
    col = c_start + cuda.blockIdx.x * 16 + tx

    # Shared-memory tile: 16x16 + one-cell apron
    smem = cuda.shared.array(
        shape=(18, 18),
        dtype=float64
    )

    sy = ty + 1
    sx = tx + 1

    # All threads participate in loading the tile.
    # Bounding-box guards are applied during output.

    max_row = u.shape[0] - 1
    max_col = u.shape[1] - 1

    valid = row >= 0 and row <= max_row and col >= 0 and col <= max_col

    if valid:
        smem[sy, sx] = u[row, col]
    else:
        smem[sy, sx] = 0.0

    # Load the four halo strips
    if ty == 0:
        if row > 0 and col >= 0 and col <= max_col:
            smem[0, sx] = u[row - 1, col]
        else:
            smem[0, sx] = 0.0

    if ty == 15:
        if row < max_row and col >= 0 and col <= max_col:
            smem[17, sx] = u[row + 1, col]
        else:
            smem[17, sx] = 0.0

    if tx == 0:
        if col > 0 and row >= 0 and row <= max_row:
            smem[sy, 0] = u[row, col - 1]
        else:
            smem[sy, 0] = 0.0

    if tx == 15:
        if col < max_col and row >= 0 and row <= max_row:
            smem[sy, 17] = u[row, col + 1]
        else:
            smem[sy, 17] = 0.0

    cuda.syncthreads()

    # Only update cells inside the requested bounding box
    if (
        row >= r_start and row <= r_end
        and col >= c_start and col <= c_end
        and row >= 1 and row < max_row
        and col >= 1 and col < max_col
    ):
        center = smem[sy, sx]
        up = smem[sy - 1, sx]
        down = smem[sy + 1, sx]
        left = smem[sy, sx - 1]
        right = smem[sy, sx + 1]

        kc = kappa[row, col]

        kup = 0.5 * (kc + kappa[row - 1, col])
        kdown = 0.5 * (kc + kappa[row + 1, col])
        kleft = 0.5 * (kc + kappa[row, col - 1])
        kright = 0.5 * (kc + kappa[row, col + 1])

        diffusion_y = (
            kdown * (down - center)
            - kup * (center - up)
        ) / (dy * dy)

        diffusion_x = (
            kright * (right - center)
            - kleft * (center - left)
        ) / (dx * dx)

        reaction = -LAMBDA_DAMP * center * center * center

        u_new[row, col] = center + dt * (
            diffusion_x
            + diffusion_y
            + reaction
            + source[row, col]
        )


def launch_stencil(
    d_u, d_next, d_kappa, d_source,
    dx, dy, dt,
    r_start, r_end, c_start, c_end,
    stream
):
    rows = r_end - r_start + 1
    cols = c_end - c_start + 1

    blocks = (
        (cols + 15) // 16,
        (rows + 15) // 16
    )

    tiled_stencil_kernel[blocks, (16, 16), stream](
        d_u, d_next, d_kappa, d_source,
        dx, dy, dt,
        r_start, r_end, c_start, c_end
    )

def test_stencil_step(context, d_u, d_kappa, d_source):
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    # Conservative explicit time step
    dt = 0.25 * dx * dx / 0.24

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    d_next = cuda.device_array_like(d_u)

    # Exchange halos before evaluating the stencil
    exchange_halos_async(
        context, d_u, stream_boundary
    )

    # First compute the deep interior
    launch_stencil(
        d_u, d_next, d_kappa, d_source,
        dx, dy, dt,
        2, ny - 1, 2, nx - 1,
        stream_interior
    )

    # Then compute four non-overlapping boundary strips
    launch_stencil(
        d_u, d_next, d_kappa, d_source,
        dx, dy, dt,
        1, 1, 1, nx,
        stream_boundary
    )

    launch_stencil(
        d_u, d_next, d_kappa, d_source,
        dx, dy, dt,
        ny, ny, 1, nx,
        stream_boundary
    )

    launch_stencil(
        d_u, d_next, d_kappa, d_source,
        dx, dy, dt,
        2, ny - 1, 1, 1,
        stream_boundary
    )

    launch_stencil(
        d_u, d_next, d_kappa, d_source,
        dx, dy, dt,
        2, ny - 1, nx, nx,
        stream_boundary
    )

    stream_interior.synchronize()
    stream_boundary.synchronize()

    # Copy only for testing the result
    result = d_next.copy_to_host()

    local_min = np.min(result[1:ny+1, 1:nx+1])
    local_max = np.max(result[1:ny+1, 1:nx+1])

    if not np.all(np.isfinite(result[1:ny+1, 1:nx+1])):
        raise RuntimeError("Stencil produced invalid values")

    print(
        f"Rank {context['rank']}: "
        f"stencil completed, "
        f"min={local_min:.8f}, "
        f"max={local_max:.8f}",
        flush=True
    )


# ============================================================
# TASK 4: NON-BLOCKING HALO EXCHANGE
# ============================================================

def start_halo_exchange(context, d_u, stream_boundary):
    cart = context["cart_comm"]
    neighbors = context["neighbors"]

    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    send_dev = {}
    recv_dev = {}
    send_host = {}
    recv_host = {}
    requests = []

    directions = {
        0: (neighbors[0], nx),
        1: (neighbors[1], nx),
        2: (neighbors[2], ny),
        3: (neighbors[3], ny),
    }

    # Stage A: GPU packing and device-to-host copies
    for direction, (neighbor, count) in directions.items():
        if neighbor == MPI.PROC_NULL:
            continue

        send_dev[direction] = cuda.device_array(
            count, dtype=np.float64
        )
        recv_dev[direction] = cuda.device_array(
            count, dtype=np.float64
        )

        send_host[direction] = cuda.pinned_array(
            count, dtype=np.float64
        )
        recv_host[direction] = cuda.pinned_array(
            count, dtype=np.float64
        )

        blocks = (count + 255) // 256

        pack_halos_kernel[
            blocks, 256, stream_boundary
        ](
            d_u, send_dev[direction],
            direction, ny, nx
        )

        send_dev[direction].copy_to_host(
            send_host[direction],
            stream=stream_boundary
        )

    # Host buffers must be ready before MPI reads them.
    stream_boundary.synchronize()

    # Stage B: Post non-blocking receives
    for direction, (neighbor, count) in directions.items():
        if neighbor == MPI.PROC_NULL:
            continue

        requests.append(
            cart.Irecv(
                recv_host[direction],
                source=neighbor,
                tag=direction ^ 1
            )
        )

    # Post non-blocking sends
    for direction, (neighbor, count) in directions.items():
        if neighbor == MPI.PROC_NULL:
            continue

        requests.append(
            cart.Isend(
                send_host[direction],
                dest=neighbor,
                tag=direction
            )
        )

    return {
        "requests": requests,
        "recv_dev": recv_dev,
        "recv_host": recv_host,
        "send_dev": send_dev,
        "send_host": send_host,
        "directions": directions
    }


def finish_halo_exchange(
    context, d_u, stream_boundary, exchange
):
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    # Stage C: Wait for MPI while GPU interior
    # work may already be running.
    MPI.Request.Waitall(exchange["requests"])

    for direction, (neighbor, count) in exchange["directions"].items():
        if neighbor == MPI.PROC_NULL:
            continue

        exchange["recv_dev"][direction].copy_to_device(
            exchange["recv_host"][direction],
            stream=stream_boundary
        )

        blocks = (count + 255) // 256

        unpack_halos_kernel[
            blocks, 256, stream_boundary
        ](
            d_u,
            exchange["recv_dev"][direction],
            direction, ny, nx
        )

    # Ensure unpack is finished before boundary stencil.
    stream_boundary.synchronize()


def step_overlapped(
    context, d_u, d_next, d_kappa, d_source,
    dx, dy, dt, stream_boundary, stream_interior
):
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    # Begin asynchronous MPI communication
    exchange = start_halo_exchange(
        context, d_u, stream_boundary
    )

    # Compute deep interior without waiting for MPI
    launch_stencil(
        d_u, d_next, d_kappa, d_source,
        dx, dy, dt,
        2, ny - 1, 2, nx - 1,
        stream_interior
    )

    # Complete MPI exchange and update ghost cells
    finish_halo_exchange(
        context, d_u, stream_boundary, exchange
    )

    # Compute four boundary strips
    for bounds in [
        (1, 1, 1, nx),
        (ny, ny, 1, nx),
        (2, ny - 1, 1, 1),
        (2, ny - 1, nx, nx)
    ]:
        launch_stencil(
            d_u, d_next, d_kappa, d_source,
            dx, dy, dt,
            *bounds,
            stream_boundary
        )

    # Both streams must finish before swapping arrays
    stream_interior.synchronize()
    stream_boundary.synchronize()

    return d_next, d_u



def test_overlapped_step(context, d_u, d_kappa, d_source):
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    dt = 0.8 * dx * dx / (4.0 * 0.24)

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    # Initialize output including zero-valued physical ghost cells
    d_next = cuda.to_device(
        np.zeros((ny + 2, nx + 2), dtype=np.float64)
    )

    d_result, d_old = step_overlapped(
        context,
        d_u, d_next, d_kappa, d_source,
        dx, dy, dt,
        stream_boundary, stream_interior
    )

    result = d_result.copy_to_host()
    owned = result[1:ny + 1, 1:nx + 1]

    if not np.all(np.isfinite(owned)):
        raise RuntimeError("Invalid overlapped stencil values")

    print(
        f"Rank {context['rank']}: "
        f"OVERLAPPED STEP SUCCESSFUL, "
        f"min={owned.min():.8f}, "
        f"max={owned.max():.8f}",
        flush=True
    )


# ============================================================
# TASK 5: SERIALIZED SOLVER, METRICS AND BENCHMARK
# ============================================================

import time
import re


def step_serialized(
    context, d_u, d_next, d_kappa, d_source,
    dx, dy, dt, stream_boundary
):
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    # Complete communication before computation
    exchange_halos_async(context, d_u, stream_boundary)

    # Use only one CUDA stream
    bounds_list = [
        (2, ny - 1, 2, nx - 1),
        (1, 1, 1, nx),
        (ny, ny, 1, nx),
        (2, ny - 1, 1, 1),
        (2, ny - 1, nx, nx)
    ]

    for bounds in bounds_list:
        launch_stencil(
            d_u, d_next, d_kappa, d_source,
            dx, dy, dt,
            *bounds,
            stream_boundary
        )

    stream_boundary.synchronize()

    return d_next, d_u


def distributed_metrics(context, d_u, d_previous, dx, dy):
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    current = d_u.copy_to_host()[1:ny+1, 1:nx+1]

    local_energy = np.array(
        [np.sum(current) * dx * dy],
        dtype=np.float64
    )
    global_energy = np.zeros(1, dtype=np.float64)

    if d_previous is None:
        local_residual_sq = 0.0
    else:
        previous = d_previous.copy_to_host()[1:ny+1, 1:nx+1]
        difference = current - previous
        local_residual_sq = (
            np.sum(difference * difference) * dx * dy
        )

    local_residual = np.array(
        [local_residual_sq], dtype=np.float64
    )
    global_residual = np.zeros(1, dtype=np.float64)

    cart = context["cart_comm"]

    cart.Allreduce(
        local_energy, global_energy, op=MPI.SUM
    )
    cart.Allreduce(
        local_residual, global_residual, op=MPI.SUM
    )

    return (
        float(global_energy[0]),
        float(np.sqrt(global_residual[0]))
    )


def run_mode(
    mode, context, initial_host,
    d_kappa, d_source, dx, dy, dt
):
    rank = context["rank"]
    cart = context["cart_comm"]

    d_u = cuda.to_device(initial_host.copy())
    d_next = cuda.to_device(np.zeros_like(initial_host))

    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    # Compile and warm up both execution paths
    warm_u = cuda.to_device(initial_host.copy())
    warm_next = cuda.to_device(np.zeros_like(initial_host))

    if mode == "serialized":
        step_serialized(
            context, warm_u, warm_next,
            d_kappa, d_source, dx, dy, dt,
            stream_boundary
        )
    else:
        step_overlapped(
            context, warm_u, warm_next,
            d_kappa, d_source, dx, dy, dt,
            stream_boundary, stream_interior
        )

    cuda.synchronize()
    cart.Barrier()

    start_event = cuda.event(timing=True)
    end_event = cuda.event(timing=True)

    start_event.record(stream_boundary)
    start_wall = time.perf_counter()

    energy_trace = []

    energy, residual = distributed_metrics(
        context, d_u, None, dx, dy
    )
    energy_trace.append((0, energy, residual))

    if rank == 0:
        print(
            f"{mode} step=0 "
            f"energy={energy:.12e} "
            f"residual={residual:.12e}",
            flush=True
        )

    for step in range(1, STEPS + 1):
        if mode == "serialized":
            d_u, d_previous = step_serialized(
                context, d_u, d_next,
                d_kappa, d_source, dx, dy, dt,
                stream_boundary
            )
        else:
            d_u, d_previous = step_overlapped(
                context, d_u, d_next,
                d_kappa, d_source, dx, dy, dt,
                stream_boundary, stream_interior
            )

        d_next = d_previous

        if step % 100 == 0:
            energy, residual = distributed_metrics(
                context, d_u, d_previous, dx, dy
            )
            energy_trace.append((step, energy, residual))

            if rank == 0:
                print(
                    f"{mode} step={step} "
                    f"energy={energy:.12e} "
                    f"residual={residual:.12e}",
                    flush=True
                )

    cuda.synchronize()
    end_event.record(stream_boundary)
    end_event.synchronize()

    local_wall_ms = (
        time.perf_counter() - start_wall
    ) * 1000.0

    # Slowest MPI rank determines distributed runtime
    wall_ms = cart.allreduce(
        local_wall_ms, op=MPI.MAX
    )

    # CUDA event timing is GPU-stream telemetry,
    # not complete MPI wall-clock runtime.
    local_gpu_ms = cuda.event_elapsed_time(
        start_event, end_event
    )
    gpu_ms = cart.allreduce(
        local_gpu_ms, op=MPI.MAX
    )

    final_host = d_u.copy_to_host()

    return {
        "final": final_host,
        "wall_ms": float(wall_ms),
        "gpu_event_ms": float(gpu_ms),
        "energy_trace": energy_trace
    }


def ptx_audit():
    signatures = tiled_stencil_kernel.signatures

    if not signatures:
        raise RuntimeError("Stencil kernel was not compiled")

    ptx = tiled_stencil_kernel.inspect_asm(signatures[0])

    counts = {
        "st.shared.f64": len(
            re.findall(r"\bst\.shared\.f64\b", ptx)
        ),
        "ld.shared.f64": len(
            re.findall(r"\bld\.shared\.f64\b", ptx)
        ),
        "bar.sync": len(
            re.findall(r"\bbar\.sync\b", ptx)
        )
    }

    passed = all(value > 0 for value in counts.values())

    return counts, passed


def run_solver(github_user):
    context = init_distributed_context(github_user)
    cart = context["cart_comm"]
    rank = context["rank"]

    d_u, d_kappa, d_source = initialize_subdomains(context)

    initial_host = d_u.copy_to_host()

    dx = 1.0 / (GLOBAL_NX + 1)
    dy = 1.0 / (GLOBAL_NY + 1)

    local_kmax = np.array(
        [np.max(d_kappa.copy_to_host())],
        dtype=np.float64
    )
    global_kmax = np.zeros(1, dtype=np.float64)

    cart.Allreduce(
        local_kmax, global_kmax, op=MPI.MAX
    )

    kmax = float(global_kmax[0])

    dt_max = dx * dx / (4.0 * kmax)
    dt = 0.8 * dt_max

    if rank == 0:
        print("=" * 50)
        print("STARTING 400-STEP BENCHMARK")
        print("MPI RANKS:", context["size"])
        print("INSTANCE_SEED:", context["instance_seed"])
        print("KAPPA_MAX:", kmax)
        print("DT_MAX:", dt_max)
        print("DT:", dt)
        print("=" * 50, flush=True)

    # Run serialized solver
    serialized = run_mode(
        "serialized",
        context, initial_host,
        d_kappa, d_source, dx, dy, dt
    )

    cart.Barrier()

    # Run overlapped solver
    overlapped = run_mode(
        "overlapped",
        context, initial_host,
        d_kappa, d_source, dx, dy, dt
    )

    # Compare owned cells only
    ny = GLOBAL_NY // 2
    nx = GLOBAL_NX // 2

    local_diff = np.array([
        np.max(np.abs(
            serialized["final"][1:ny+1, 1:nx+1]
            - overlapped["final"][1:ny+1, 1:nx+1]
        ))
    ], dtype=np.float64)

    global_diff = np.zeros(1, dtype=np.float64)

    cart.Allreduce(
        local_diff, global_diff, op=MPI.MAX
    )

    parity_max_diff = float(global_diff[0])
    parity_passed = parity_max_diff < 1e-12

    ptx_counts, ptx_passed = ptx_audit()

    # A rank-wide PTX audit
    all_ptx_passed = cart.allreduce(
        int(ptx_passed), op=MPI.MIN
    ) == 1

    if rank == 0:
        serialized_ms = serialized["wall_ms"]
        overlapped_ms = overlapped["wall_ms"]

        speedup = serialized_ms / overlapped_ms

        manifest = {
            "github_username": github_user,
            "gpu_uuid": context["gpu_uuid"],
            "instance_seed": context["instance_seed"],
            "mpi_ranks": context["size"],
            "cartesian_dims": [2, 2],
            "steps": STEPS,
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "kappa_max": kmax,
            "dt_max": dt_max,
            "dt": dt,
            "serialized_ms": serialized_ms,
            "overlapped_ms": overlapped_ms,
            "speedup": speedup,
            "serialized_gpu_event_ms": serialized["gpu_event_ms"],
            "overlapped_gpu_event_ms": overlapped["gpu_event_ms"],
            "parity_max_diff": parity_max_diff,
            "parity_passed": parity_passed,
            "ptx_counts": ptx_counts,
            "ptx_audit_passed": all_ptx_passed,
            "energy_trace": overlapped["energy_trace"]
        }

        with open(
            "submission_manifest.json", "w"
        ) as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "=" * 50)
        print("FINAL BENCHMARK RESULTS")
        print("=" * 50)
        print(f"serialized_ms: {serialized_ms:.3f}")
        print(f"overlapped_ms: {overlapped_ms:.3f}")
        print(f"speedup: {speedup:.4f}")
        print(f"parity_max_diff: {parity_max_diff:.12e}")
        print(f"parity_passed: {parity_passed}")
        print(f"PTX counts: {ptx_counts}")
        print(f"ptx_audit_passed: {all_ptx_passed}")

        print("\nManifest: submission_manifest.json")

        if parity_passed and all_ptx_passed:
            print("ALL NUMERICAL AND PTX CHECKS PASSED")
        else:
            print("WARNING: SOME CHECKS FAILED")


if __name__ == "__main__":
    github_user = sys.argv[1]
    run_solver(github_user)
