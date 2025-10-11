[[Event Loop]]

`asyncio`是Python 3.4版本引入的标准库，直接内置了对异步IO的支持。

`asyncio`的编程模型就是一个消息循环。`asyncio`模块内部实现了`EventLoop`，把需要执行的协程扔到`EventLoop`中执行，就实现了异步IO。

用`asyncio`提供的`@asyncio.coroutine`可以把一个`generator`标记为`coroutine`类型，然后在`coroutine`内部用`yield from`调用另一个`coroutine`实现异步操作。

为了简化并更好地标识异步IO，从Python 3.5开始引入了新的语法`async`和`await`，可以让`coroutine`的代码更简洁易读。

用`asyncio`实现`Hello world`代码如下：

```python
import asyncio

async def hello():
    print("Hello world!")
    # 异步调用asyncio.sleep(1):
    await asyncio.sleep(1)
    print("Hello again!")

asyncio.run(hello())
```

`async`把一个函数变成`coroutine`类型，然后，我们就把这个`async`函数扔到`asyncio.run()`中执行。执行结果如下：

```plain
Hello!
(等待约1秒)
Hello again!
```

`hello()`会首先打印出`Hello world!`，然后，`await`语法可以让我们方便地调用另一个`async`函数。由于`asyncio.sleep()`也是一个`async`函数，所以线程不会等待`asyncio.sleep()`，而是直接中断并执行下一个消息循环。当`asyncio.sleep()`返回时，就接着执行下一行语句。

把`asyncio.sleep(1)`看成是一个耗时1秒的IO操作，在此期间，主线程并未等待，而是去执行`EventLoop`中其他可以执行的`async`函数了，因此可以实现并发执行。

上述`hello()`还没有看出并发执行的特点，我们改写一下，让两个`hello()`同时并发执行：

```python
# 传入name参数:
async def hello(name):
    # 打印name和当前线程:
    print("Hello %s! (%s)" % (name, threading.current_thread))
    # 异步调用asyncio.sleep(1):
    await asyncio.sleep(1)
    print("Hello %s again! (%s)" % (name, threading.current_thread))
    return name
```

用`asyncio.gather()`同时调度多个`async`函数：

```python
async def main():
    L = await asyncio.gather(hello("Bob"), hello("Alice"))
    print(L)

asyncio.run(main())
```

执行结果如下：

```plain
Hello Bob! (<function current_thread at 0x10387d260>)
Hello Alice! (<function current_thread at 0x10387d260>)
(等待约1秒)
Hello Bob again! (<function current_thread at 0x10387d260>)
Hello Alice again! (<function current_thread at 0x10387d260>)
['Bob', 'Alice']
```

从结果可知，用`asyncio.run()`执行`async`函数，所有函数均由同一个线程执行。两个`hello()`是并发执行的，并且可以拿到`async`函数执行的结果（即`return`的返回值）。

如果把`asyncio.sleep()`换成真正的IO操作，则多个并发的IO操作实际上可以由一个线程并发执行。

我们用`asyncio`的异步网络连接来获取sina、sohu和163的网站首页：

```python
import asyncio

async def wget(host):
    print(f"wget {host}...")
    # 连接80端口:
    reader, writer = await asyncio.open_connection(host, 80)
    # 发送HTTP请求:
    header = f"GET / HTTP/1.0\r\nHost: {host}\r\n\r\n"
    writer.write(header.encode("utf-8"))
    await writer.drain()

    # 读取HTTP响应:
    while True:
        line = await reader.readline()
        if line == b"\r\n":
            break
        print("%s header > %s" % (host, line.decode("utf-8").rstrip()))
    # Ignore the body, close the socket
    writer.close()
    await writer.wait_closed()
    print(f"Done {host}.")

async def main():
    await asyncio.gather(wget("www.sina.com.cn"), wget("www.sohu.com"), wget("www.163.com"))

asyncio.run(main())
```

执行结果如下：

```plain
wget www.sohu.com...
wget www.sina.com.cn...
wget www.163.com...
(等待一段时间)
(打印出sohu的header)
www.sohu.com header > HTTP/1.1 200 OK
www.sohu.com header > Content-Type: text/html
...
(打印出sina的header)
www.sina.com.cn header > HTTP/1.1 200 OK
www.sina.com.cn header > Date: Wed, 20 May 2015 04:56:33 GMT
...
(打印出163的header)
www.163.com header > HTTP/1.0 302 Moved Temporarily
www.163.com header > Server: Cdn Cache Server V2.0
...
```

可见3个连接由一个线程并发执行3个`async`函数完成。

### 小结

`asyncio`提供了完善的异步IO支持，用`asyncio.run()`调度一个`coroutine`；

在一个`async`函数内部，通过`await`可以调用另一个`async`函数，这个调用看起来是串行执行的，但实际上是由`asyncio`内部的消息循环控制；

在一个`async`函数内部，通过`await asyncio.gather()`可以并发执行若干个`async`函数。





## asyncio api

### 一、 运行 `asyncio` 程序 (高层 API)



这是启动和运行异步应用的入口。

| API 方法                                 | 作用说明                                                     |
| ---------------------------------------- | ------------------------------------------------------------ |
| **`asyncio.run(coro, \*, debug=False)`** | **执行顶层协程的入口函数**。这是启动一个 `asyncio` 程序的**首选方式**。它会自动创建和关闭事件循环，处理任务的完成和异常。你可以把它看作是异步世界的 `main()` 函数启动器。 |

------



### 二、 创建和管理任务 (高层 API)



任务（Task）是并发执行协程的核心。

| API 方法                                             | 作用说明                                                     |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| **`asyncio.create_task(coro, \*, name=None)`**       | **将一个协程包装成一个任务（Task）并提交到事件循环中尽快执行**。这是实现并发的关键：它不会阻塞当前代码，而是让协程“在后台”独立运行。 |
| **`asyncio.gather(\*aws, return_exceptions=False)`** | **并发运行多个可等待对象（协程、任务或 Future）**，并**按输入顺序**收集它们的结果。如果任何一个任务失败（抛出异常），默认情况下 `gather` 会立即将该异常传播出去，并取消其他未完成的任务。 |
| **`asyncio.wait_for(aw, timeout)`**                  | **为一个可等待对象设置超时**。如果在 `timeout` 秒内任务没有完成，它会引发一个 `asyncio.TimeoutError` 并取消该任务。这对于防止任务无限期阻塞非常有用。 |
| **`asyncio.shield(aw)`**                             | **保护一个可等待对象不被取消**。如果包含 `shield()` 的外层任务被取消，被保护的任务会继续在后台运行。这在需要确保某个关键操作（如数据库写入）必须完成时非常重要。 |
| **`asyncio.sleep(delay, result=None)`**              | **非阻塞地暂停当前协程指定的秒数**。在暂停期间，事件循环会去执行其他任务。这是模拟I/O等待或实现定时任务的常用方法。 |

------



### 三、 同步原语 (Synchronization Primitives)



当多个并发任务需要协调工作或访问共享资源时，就需要使用同步原语，这与多线程编程中的概念类似。

| API 方法                       | 作用说明                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| **`asyncio.Lock`**             | **异步锁**。用于保护共享资源，确保在任何时候只有一个任务可以访问该资源。一个任务在进入临界区代码前 `await lock.acquire()`，在离开时调用 `lock.release()`。 |
| **`asyncio.Event`**            | **异步事件**。一个任务可以等待（`await event.wait()`）一个内部标志位被设置。另一个任务可以通过 `event.set()` 将标志位置为 `True`，从而唤醒所有正在等待的任务。`event.clear()` 则可以将标志位重置为 `False`。 |
| **`asyncio.Condition`**        | **异步条件变量**。比 `Lock` 更复杂的同步机制。一个任务可以等待某个条件成立（`await cond.wait()`），而另一个任务在满足条件后可以通知（`cond.notify()`）一个或所有等待的任务。它总是与一个底层锁相关联。 |
| **`asyncio.Semaphore`**        | **异步信号量**。用于限制能同时访问某个资源的并发任务数量。例如，限制对某个 API 的并发调用不能超过10次。 |
| **`asyncio.BoundedSemaphore`** | 与 `Semaphore` 类似，但它会确保 `release()` 的调用次数不会超过 `acquire()` 的次数，防止程序逻辑错误。 |
| **`asyncio.Queue`**            | **异步队列**。一个先进先出（FIFO）的数据结构，用于在多个并发的生产者和消费者任务之间安全地传递数据。`await queue.put(item)` 用于放入数据，`await queue.get()` 用于取出数据。 |

------



### 四、 流 (Streams) - 用于网络 I/O (高层 API)



Streams API 是处理网络连接（如 TCP）的推荐方式，它提供了简单易用的读写接口。

| API 方法                                                    | 作用说明                                                     |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| **`asyncio.open_connection(host, port)`**                   | **建立一个 TCP 客户端连接**。成功后返回一对 `(reader, writer)` 对象，用于从套接字读取和写入数据。 |
| **`asyncio.start_server(client_connected_cb, host, port)`** | **启动一个 TCP 服务器**。`client_connected_cb` 是一个回调函数，每当有新的客户端连接时，它会被调用，并接收一对 `(reader, writer)` 参数。 |
| **`StreamReader`**                                          | **流读取器对象**。提供 `read()`, `readline()`, `readexactly()` 等协程方法来异步地从流中读取数据。 |
| **`StreamWriter`**                                          | **流写入器对象**。提供 `write()`, `writelines()` 方法来写入数据，以及 `drain()` 协程来等待底层缓冲区排空，`close()` 来关闭连接。 |

------



### 五、 低层 API (不推荐普通用户直接使用)



这些 API 提供了对事件循环更精细的控制，主要供框架和库的开发者使用。直接使用它们会使代码更复杂且容易出错。

| API 方法                                           | 作用说明                                                     |
| -------------------------------------------------- | ------------------------------------------------------------ |
| **`asyncio.get_running_loop()`**                   | 获取当前正在运行的事件循环对象。                             |
| **`loop.create_future()`**                         | 创建一个 `asyncio.Future` 对象。Future 对象代表一个未来某个时刻才会完成的异步操作的结果。任务（Task）就是 Future 的一种。 |
| **`loop.call_soon(callback, \*args)`**             | 安排一个普通函数（非协程）在事件循环的下一次迭代中尽快被调用。 |
| **`loop.call_at(when, callback, \*args)`**         | 安排一个普通函数在事件循环的指定时间点 `when` 被调用。       |
| **`loop.run_in_executor(executor, func, \*args)`** | 在一个独立的线程池或进程池 (`executor`) 中**运行一个阻塞的同步函数 `func`**，并返回一个可等待的 Future。这是在异步代码中调用**阻塞I/O或CPU密集型函数**的正确方式，可以避免阻塞整个事件循环。 |

**总结:**

对于日常开发，你应该将注意力集中在**高层 API** 上。`asyncio.run()` 是你的起点，`asyncio.create_task()` 和 `asyncio.gather()` 是实现并发的核心，而 `asyncio.sleep()` 和同步原语则用于控制和协调任务。当你需要进行网络编程时，Streams API 是你的最佳选择。只有在与阻塞代码交互时，你才可能需要考虑低层的 `loop.run_in_executor()`。