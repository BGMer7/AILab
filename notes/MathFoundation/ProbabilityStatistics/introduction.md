- 想过滤垃圾邮件，不具备概率论中的[贝叶斯思维](https://www.zhihu.com/search?q=%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%80%9D%E7%BB%B4&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)恐怕不行；
- 想试着进行一段语音识别，则必须要理解随机过程中的[隐马尔科夫模型](https://www.zhihu.com/search?q=%E9%9A%90%E9%A9%AC%E5%B0%94%E7%A7%91%E5%A4%AB%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)；
- 想通过观察到的样本推断出某类对象的总体特征，估计理论和[大数定理](https://www.zhihu.com/search?q=%E5%A4%A7%E6%95%B0%E5%AE%9A%E7%90%86&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)的思想必须建立；
- 在统计推断过程中，要理解广泛采用的近似采样方法，蒙特卡洛方法以及马尔科夫过程的稳态也得好好琢磨；
- 想从文本中提取出我们想要的名称实体，[概率图模型](https://www.zhihu.com/search?q=%E6%A6%82%E7%8E%87%E5%9B%BE%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)也得好好了解。

作者：石溪  
链接：https://zhuanlan.zhihu.com/p/87438632  
来源：知乎  
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。  


在我看来，概率统计的核心部分共有以下六大部分，纵贯了概率论、统计以及随机过程中最核心的主线内容：

**第 1 部分：概率思想。**我们首先从[条件概率](https://www.zhihu.com/search?q=%E6%9D%A1%E4%BB%B6%E6%A6%82%E7%8E%87&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)和贝叶斯方法入手，阐明条件、独立、相关等基本概念，掌握联合、边缘的计算方法，我们将一起构建起认知世界的[概率思维](https://www.zhihu.com/search?q=%E6%A6%82%E7%8E%87%E6%80%9D%E7%BB%B4&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)体系。

**第 2 部分：随机变量。**我们将重点介绍随机变量主干内容，从单一随机变量的分布过渡到多元随机变量的分析，最后重点阐述大数定理和中心极限定理，并初步接触[蒙特卡洛方法](https://www.zhihu.com/search?q=%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B%E6%96%B9%E6%B3%95&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)，和读者一起建立重要的极限思维。

**第 3 部分：统计推断。**这部分我们关注的是如何通过部分的样本集合推断出我们关心的总体特征，这在现实世界中非常重要。在[参数估计](https://www.zhihu.com/search?q=%E5%8F%82%E6%95%B0%E4%BC%B0%E8%AE%A1&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)的思想方法基础上，我们重点关注[极大似然估计](https://www.zhihu.com/search?q=%E6%9E%81%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)和贝叶斯估计这两种方法。

**第 4 部分：随机过程。**我们将关注由一组随机变量构成的集合，即随机过程。股票的波动、语音信号、视频信号、[布朗运动](https://www.zhihu.com/search?q=%E5%B8%83%E6%9C%97%E8%BF%90%E5%8A%A8&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)等都是随机过程在现实世界中的实例。我们在随机过程的基本概念之上，将重点分析[马尔科夫链](https://www.zhihu.com/search?q=%E9%A9%AC%E5%B0%94%E7%A7%91%E5%A4%AB%E9%93%BE&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)，梳理其由静到动的演变，探索变化的过程和不变的稳态。

**第 5 部分：[采样理论](https://www.zhihu.com/search?q=%E9%87%87%E6%A0%B7%E7%90%86%E8%AE%BA&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)。**我们将重点关注如何获取服从目标分布的近似采样方法，从基本的接受-拒绝采样入手，逐渐深入到马尔科夫链-蒙特卡洛方法，通过动态的过程进一步深化对随机过程、随机理论以及[极限思想](https://www.zhihu.com/search?q=%E6%9E%81%E9%99%90%E6%80%9D%E6%83%B3&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)的理解。

**第 6 部分：概率模型。**这里我们将介绍概率图模型中的一种典型模型：隐马尔科夫模型，熟悉状态序列的[概率估计](https://www.zhihu.com/search?q=%E6%A6%82%E7%8E%87%E4%BC%B0%E8%AE%A1&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A%2287438632%22%7D)和状态解码的基本方法，为后续学习的概率图模型打好基础。