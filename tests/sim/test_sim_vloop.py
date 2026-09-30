# tests/sim/test_sim_vloop.py
import asyncio
import time

from sim.vloop import VirtualTimeLoop, run_virtual


def test_sleep_uses_virtual_time_and_is_fast():
    async def main():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await asyncio.sleep(90)
        return loop.time() - t0

    start = time.perf_counter()
    assert abs(run_virtual(main()) - 90) < 1e-9
    assert time.perf_counter() - start < 2


def test_timers_fire_in_order():
    order = []

    async def worker(name, delay):
        await asyncio.sleep(delay)
        order.append((name, round(asyncio.get_running_loop().time(), 6)))

    async def main():
        await asyncio.gather(worker("b", 2.0), worker("a", 0.5), worker("c", 3.25))

    run_virtual(main())
    assert order == [("a", 0.5), ("b", 2.0), ("c", 3.25)]


def test_wait_for_timeout_is_virtual():
    async def main():
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(asyncio.sleep(100), timeout=5)
        except asyncio.TimeoutError:
            return loop.time()

    assert abs(run_virtual(main()) - 5) < 1e-9


def test_executor_work_is_not_skipped():
    async def main():
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, time.sleep, 0.05)
        t0 = loop.time()
        await fut
        return loop.time() - t0

    elapsed = run_virtual(main())
    assert 0.04 <= elapsed < 1.0


def test_charge_compute_advances_clock():
    async def main():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        end = time.perf_counter() + 0.05
        while time.perf_counter() < end:
            pass
        await asyncio.sleep(0)
        return loop.time() - t0

    assert run_virtual(main(), charge_compute=True) >= 0.04
    assert run_virtual(main(), charge_compute=False) == 0.0


def test_wall_cap_callback():
    hit = []
    loop = VirtualTimeLoop(wall_cap_s=0.0, on_wall_cap=lambda: hit.append(True))

    async def main():
        await asyncio.sleep(1)
        await asyncio.sleep(1)

    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    assert hit == [True] and loop.wall_cap_hit
