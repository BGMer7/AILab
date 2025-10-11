import asyncio
import time
import datetime


# 1. 定义协程函数
async def download_file_async(filename: str):
    print(f"[{datetime.datetime.now().time()}] 开始下载 {filename}...")
    # 2. 使用 await 暂停，并切换任务
    await asyncio.sleep(3)  # 非阻塞的sleep，会把控制权交还给事件循环
    print(f"[{datetime.datetime.now().time()}] {filename} 下载完成.")


# 3. 定义一个主入口协程
async def main_async():
    start_time = time.time()

    # 4. 创建任务，让它们并发执行
    task1 = asyncio.create_task(download_file_async("file1.txt"))
    task2 = asyncio.create_task(download_file_async("file2.txt"))

    # 等待所有任务完成
    await task1
    await task2
    # (更优雅的写法是: await asyncio.gather(task1, task2))

    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")


# 5. 启动事件循环并运行主协程
asyncio.run(main_async())
