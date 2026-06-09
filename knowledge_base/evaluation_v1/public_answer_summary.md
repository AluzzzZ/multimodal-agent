# question_public 批量答题质量摘要

- 题目数: `400`
- LLM provider: `openai`
- 平均置信度: `0.936`
- 中位置信度: `0.95`
- 平均回答长度: `557`
- 平均关键词覆盖度: `0.6368`
- fallback 数: `0`
- 多问题题数: `16`
- 含 `<PIC>` 回答数: `252`
- 使用分类器的题数: `0`
- 服务型问题却被主判为 manual 的题数: `0`

## 路由分布
- service: `52`
- manual: `320`
- mixed: `28`

## 风险分布
- low: `374`
- high: `6`
- medium: `20`

## Mixed 示例
- id=72 confidence=0.9 question=如何给空调遥控器安装电池？
  answer=1. 取下电池盖。  
   <PIC>  
2. 装入新的7号电池，确保电池正、负极安装正确。  
   <PIC>  
3. 重新装上电池盖。  

**操作须知**  
- 不要混用不同型号的电池，也不要将新电池与旧电池混用。  
- 不要对电池充电或拆解电池。  
- 若电池漏液，立即停止使用；皮肤或衣物接触漏液需用清水冲洗，误食漏液须彻底漱口并立即就医。

["Manual01_2", "Manual01_3"]
- id=73 confidence=0.95 question=如何安装空调遥控器支架？
  answer=安装空调遥控器支架的方法如下：

1. 选择位置  
   为保护遥控器，请将支架安装在无阳光直射的地方，同时确保位置安全且便于取用。

2. 固定支架  
   用螺丝刀将两颗螺丝拧紧，将支架固定在选定位置。  
   <PIC>

3. 放入遥控器  
   将遥控器滑入已固定的支架中。  
   <PIC>

如果在安装过程中遇到额外收费或需要上门安装服务等问题，可进一步提供订单信息进行核实。

["Manual01_4", "M
- id=99 confidence=0.95 question=如何根据洗碗机型号安装餐具篮？
  answer=根据您的问题，餐具篮的安装方式会因洗碗机型号不同而存在差异，手册中并未提供所有型号通用的详细安装步骤，但可以参考以下说明进行操作：

1. 确认您的洗碗机是否配有餐具篮  
   餐具篮属于“视型号而定”的配件，并非所有洗碗机都附带。您可以在产品包装清单或说明书中确认本机是否包含餐具篮。如果包含，它通常用于摆放叉、勺等小型餐具，以获得更干净的清洗效果。

2. 找到餐具篮的放置位置  
   在洗碗机内部结构示意中，餐具篮编号为10，一
- id=100 confidence=1.0 question=使用洗碗机前如何安装可折叠下层篮架？
  answer=根据洗碗机手册，可折叠下层篮架丝（通常为2根）已预装在机器下层碗篮上，无需额外安装。您只需在使用前根据放置需求将其展开或折叠：

1. 展开使用：架丝默认可能为立起状态，直接用于放置餐具。
2. 折叠创造大空间：如果需要放置锅、碗等大件物品，可以单独或同时将每根架丝折叠。操作方法为按下架丝上的卡扣（A），然后将架丝向下放平即可。

参考示意图：
<PIC>

如果您的碗篮上没有找到该折叠架丝，可能是您所购型号不支持此功能，具体以实物为准
- id=105 confidence=1.0 question=如何安装空气净化器的脚轮？
  answer=根据空气净化器手册，脚轮安装为可选配件，具体步骤如下：

1. 检查配件
   打开空气净化器包装盒，找到内含脚轮的小盒，里面应有 4 个脚轮和 8 颗螺丝。

2. 拆卸原有底座
   使用螺丝刀将空气净化器底部的四个原有底座卸下。

3. 安装脚轮
   将每个脚轮对准原底座孔位，用两颗螺丝固定，四个脚轮依次装好。

4. 保留替换件
   拆下的底座和螺丝建议妥善保存，以便日后恢复使用。

提醒：若滤盒内找不到脚轮盒，请联系客服核
- id=126 confidence=1.0 question=使用电钻时，如何安装单套无键夹头？
  answer=安装单套无键夹头（钻头或其它附件）请按以下步骤操作，并务必遵守安全警告：

1. 关闭工具并断开工具电源  
   <PIC>

2. 一只手握住夹头黑色套筒，另一只手固定工具。逆时针旋转套筒，直到开口足够大，可以装入所需附件。  
   <PIC>

3. 将附件插入夹头约 3/4 英寸（19 毫米），一只手顺时针旋转夹头套筒紧固，另一只手握住工具。您的工具配有自动主轴锁定机构，可单手开合夹头，但务必一只手握住夹头套筒、另一只手握住工
- id=127 confidence=1.0 question=如果要安装和拆卸电钻的电池组，需要遵循什么步骤？
  answer=根据电钻手册，安装和拆卸电池组的步骤如下，请务必先阅读安全提示：

安全警告
在调整或拆卸/安装电池组之前，请先关闭工具并取下电池组，意外启动可能导致受伤。

安装电池组
1. 确保电池组已充满电，以获得最佳使用效果。
2. 将电池组（K）对准工具手柄内的导轨。
3. 沿着手柄平稳滑入，直到电池组牢固就位，不会松脱。
<PIC>

拆卸电池组
1. 按下释放按钮（J）。
2. 将电池组从工具手柄中平稳拔出。

电池组取下后，可按照手册中
- id=130 confidence=0.95 question=电钻的三年有限保修包含哪些内容？
  answer=根据您提供的参考资料，关于电钻“三年有限保修”的具体包含内容，目前可提供以下信息：

在《电钻手册》中明确提到了“三年有限保修”这一政策，但现有参考资料并未列出该有限保修政策所覆盖的具体范围、条款和限制细节。

因此，我无法向您准确说明该项保修具体包含哪些内容。要获取最准确的信息，建议您：
1.  查阅随产品附带的《电钻手册》，其中“三年有限保修”条目下应有详细条款。
2.  访问品牌官方网站，查找该型号电钻的产品支持或保修政策详情页面
- id=131 confidence=0.99 question=购买健身追踪器后，包装盒里应该有什么？
  answer=根据产品手册，包装盒内包含以下物品：

- 配有小号表带的手表（颜色和材质仅供参考）  
  <PIC>
- 充电线  
  <PIC>
- 额外大号表带  
  <PIC>

可拆卸的其他颜色或材质表带需单独购买。

["fitness_trackers_01", "fitness_trackers_02", "fitness_trackers_03"]
- id=200 confidence=0.95 question=功能键盘的保修政策通常包含哪些内容，又不包含哪些内容？
  answer=关于功能键盘的保修政策，根据通用的售后规则，通常可以从“包含内容”和“不包含内容”两方面来理解。但请注意，具体的保修条款可能因产品型号、销售渠道或活动政策存在差异，以下为一般性说明。

1. 保修政策通常包含的内容
质保期内因产品本身质量问题导致的性能故障：若功能键盘在保修期内出现非人为造成的功能异常（如按键失灵、连接故障、灯光异常等），通常可享受免费的检测、维修或更换服务。
售后凭证的灵活核实：即使保修卡丢失，一般仍可凭借订单记录、设

## 高风险示例
- id=9 route=service confidence=0.5 question=我收到的商品少了一件，联系客服说会补发，但是过了一周还没补发！
  answer=非常理解您等待一周还没收到补发的焦急心情。您的问题属于少件到货异常，我来帮您推进处理。

为了尽快核实补发进度并推动解决，需要您配合提供以下信息：
1. 订单号
2. 签收时间
3. 您收到的全部商品照片（能看清商品数量），如有外包装照片也请一并提供（尤其快递面单信息）

我们会根据这些信息去确认补发单是否已生成、是否已发货以及当前物流状态。只要情况属实，我们一定负责帮您处理到底。[待确认：补发进度查询的具体流程需人工核实工单状态]
- id=225 route=manual confidence=0.4 question=如何使用烤箱的烤架烤盘套装？
  answer=使用烤架烤盘套装的步骤如下：

1. 确认套装组成  
   烤架烤盘套装（图6）由烤架（6a）和搪瓷容器（6b）两部分组成。  
   <PIC>

2. 放置基础烤架  
   先将标准烤架（3）放入烤箱内合适的层位，作为套装的底座支撑。  
   <PIC>

3. 组合套装  
   将烤架烤盘套装整体放置在已安装好的标准烤架（3）上。

4. 选择功能  
   该套装需配合烧烤功能使用。

5. 烹饪使用  
   烤架（6
- id=266 route=manual confidence=0.4 question=How to change the default setting of the energy saving mode?
  answer=To change the setting of the energy saving mode (Eco mode) on your phone:

1. Select [Menu] > [Phone setup] > [Eco mode], then press [OK] to confirm.
2. Select [On] or [Off], then press [OK] to confirm.
   → The setting 
- id=303 route=manual confidence=0.4 question=From three different observing views, what buttons and interfaces should this eReader have?
  answer=1. **Front View** (<PIC>)
   - Home/ESC button: cancels an operation, exits a sub-interface, or returns to the previous interface.
   - Speaker: 0.5W audio output.
   - TFT LCD Display Screen: 7" diagonal.

2. **Navigati
- id=324 route=manual confidence=0.4 question=I've heard that my jetski has cruising limitations. Please introduce them.
  answer=Based on your owner's manual, here are the cruising limitations for your jetski:

- **Familiarization and constant vigilance**  
  Take time to get accustomed to handling before attempting difficult maneuvers. <PIC> Scan
- id=363 route=manual confidence=0.4 question=How can you change the engine oil of a lawn mower?
  answer=1. 准备割草机。  
   启动发动机并让其运转约5分钟，使机油升温以便更好地排出。  
   <PIC>  
   然后停车，使放油侧略低于另一侧，以确保机油能够完全排空。脱开刀片控制开关（PTO），接合驻车制动，关闭发动机，拔下钥匙，并等待所有运动部件停止。

2. 排出旧机油。  
   在发动机下方放置一个接油盘。拆下放油塞和机油加注口盖/油尺。  
   某些机型可能需要先取下盖板（参见图片）。如有放油接头，可将其安装到机油