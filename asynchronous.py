import asyncio
import time
import random

start = time.perf_counter()

async def first(x):
    await asyncio.sleep(x)
    return f'first-{x}'

async def second(x):
    await  asyncio.sleep(x)
    return f'second-{x}'

async def run_all():
    for _ in range(3):
        arr = await asyncio.gather(first(random.randint(1,3)), second(random.randint(1,3)))
        print(arr)
    print('ran all')

# asyncio.run(third())
asyncio.run(run_all())
print(time.perf_counter() - start)

