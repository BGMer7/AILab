**IoU（Intersection over Union, 交并比）** 是目标检测中衡量预测框与真实框重叠程度的重要指标，用于评估模型在定位上的精度。

### **定义**

IoU 的数学定义为：
$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

1. **Area of Overlap**:  
   预测框（Predicted Box）与真实框（Ground Truth Box）重叠区域的面积。

2. **Area of Union**:  
   预测框与真实框合并区域的面积（即两者总覆盖区域）。

### **公式分解**
假设两个矩形框分别为 $A$ 和 $B$，其交集和并集面积可以表达为：

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

其中：
- $|A \cap B|$ 是两个框的交集区域面积。
- $|A \cup B| = |A| + |B| - |A \cap B|$ 是两个框的并集区域面积。

### **计算步骤**
1. **确定交集**：
   - 计算交集区域的左上角坐标和右下角坐标。
   - 交集宽度 $\text{inter\_width} = \max(0, x_{\text{right\_min}} - x_{\text{left\_max}})$。
   - 交集高度 $\text{inter\_height} = \max(0, y_{\text{bottom\_min}} - y_{\text{top\_max}})$。
   - 交集面积为 $\text{inter\_area} = \text{inter\_width} \times \text{inter\_height}$。

2. **确定并集**：
   - 并集面积为 $|A| + |B| - \text{inter\_area}$。

3. **计算 IoU**：
   $$
   \text{IoU} = \frac{\text{inter\_area}}{\text{union\_area}}
   $$

### **用途**

1. **目标检测中的正负样本划分**：
   - IoU 是确定预测框是否与真实框匹配的标准。
   - 通常设定阈值，如 IoU ≥ 0.5（或者 0.7）认为是正确检测，否则视为误检。

2. **模型性能评估**：
   - 高 IoU 值意味着预测框与真实框高度重叠，表示定位准确。

3. **损失函数设计**：
   - 部分目标检测算法使用 IoU 或其变种（如 GIoU、DIoU、CIoU）作为损失函数，提高定位精度。

### **代码实现 IoU**

以下是一个 Python 示例，用于计算两个矩形框的 IoU：

```python
def calculate_iou(box1, box2):
    """
    计算两个矩形框的 IoU（交并比）
    
    参数：
    box1: [x1, y1, x2, y2] 第一个框的坐标（左上角和右下角）
    box2: [x1, y1, x2, y2] 第二个框的坐标（左上角和右下角）
    
    返回：
    IoU 值
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # 计算并集区域
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# 示例框
box1 = [50, 50, 150, 150]  # 左上角 (50, 50), 右下角 (150, 150)
box2 = [100, 100, 200, 200]  # 左上角 (100, 100), 右下角 (200, 200)

iou = calculate_iou(box1, box2)
print(f"IoU: {iou:.4f}")
```

### **注意点**
1. **IoU 范围**：  
   - 值在 [0, 1] 之间。
   - 0 表示没有重叠，1 表示完全重叠。

2. **阈值选择**：  
   - 常用的 IoU 阈值为 0.5 或 0.7，用于区分正负样本。

3. **边界情况**：  
   - 如果两个框完全没有重叠，则交集面积为 0，IoU 为 0。

### **扩展：IoU 的变种**
1. **GIoU（Generalized IoU）**：
   - 考虑两个框之间的最小闭包区域，提高收敛速度。

2. **DIoU（Distance IoU）**：
   - 在 GIoU 基础上，进一步考虑预测框与真实框的中心点距离。

3. **CIoU（Complete IoU）**：
   - 同时考虑距离、重叠面积和框形状的相似性。

这些改进使得目标检测模型更关注预测框与真实框的匹配质量。