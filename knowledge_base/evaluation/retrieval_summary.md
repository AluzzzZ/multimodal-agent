# 公开题检索评测摘要

- 题目数: 400
- 明细行数: 1782
- Top-K: 8
- 是否启用 rerank: True
- 检索链路: dual_route
- 平均 Top1 分数: 0.3762
- Top1 分数中位数: 0.3905
- 平均关键词覆盖度: 0.1522
- 平均 Recall@5: 0.9325
- 平均 Recall@3: 0.9325

## 命中质量分布

- high: 23
- medium: 20
- low: 330
- empty: 27

## 分层召回率（按路由类型）

| 路由 | 数量 | Recall@5 | Recall@3 | Top1 均分 | 高质量占比 | 中质量占比 | 低质量占比 |
|------|------|----------|----------|-----------|------------|------------|------------|
| manual | 320 | 0.9156 | 0.9156 | 0.3274 | 0.0 | 0.0094 | 0.9062 |
| mixed | 24 | 1.0 | 1.0 | 0.4369 | 0.0 | 0.0833 | 0.9167 |
| service | 56 | 1.0 | 1.0 | 0.6291 | 0.4107 | 0.2679 | 0.3214 |

## 候选来源统计（按最终路由）

| 路由 | service 结果数 | manual 结果数 | unknown 结果数 |
|------|----------------|---------------|----------------|
| manual | 0 | 1396 | 0 |
| mixed | 60 | 114 | 0 |
| service | 212 | 0 | 0 |

## 高频命中手册

- 摩托艇手册: 224
- 烤箱手册: 152
- 空调手册: 142
- 发电机手册: 125
- 相机手册: 116
- 健身单车手册: 97
- 洗碗机手册: 91
- 健身追踪器手册: 76
- 吹风机手册: 67
- 空气净化器手册: 55
- 电钻手册: 54
- 冰箱手册: 52
- 可编程温控器手册: 51
- 水泵手册: 46
- 蓝牙激光鼠标手册: 43
- 功能键盘手册: 40
- VR头显手册: 33
- 蒸汽清洁机手册: 20
- 儿童电动摩托车手册: 15
- 人体工学椅手册: 11

## 低命中示例（前20条）

- id=11 recall@5=1.0 top1=0.2687 coverage=0.25 route=service manuals= question=你们的商品保质期有问题，我收到的商品还有1个月就过期了！
- id=13 recall@5=1.0 top1=0.2473 coverage=0.0 route=service manuals= question=你们的商品存在质量问题，我使用了一次就坏了，联系客服没人管！
- id=25 recall@5=1.0 top1=0.2317 coverage=0.0 route=service manuals= question=请问你们家支持以旧换新服务吗？
- id=26 recall@5=1.0 top1=0.2186 coverage=0.3333 route=service manuals= question=请问你们的智能客服能解答哪些问题？ / 智能客服解答不了的问题，怎么办？
- id=35 recall@5=1.0 top1=0.4753 coverage=0.0 route=service manuals= question=想换成其他款式，能换货吗？
- id=36 recall@5=1.0 top1=0.2444 coverage=0.0 route=service manuals= question=请问商品的生产日期是什么时候？
- id=40 recall@5=1.0 top1=0.4851 coverage=0.3333 route=service manuals= question=我想给商品更换成更大的尺寸，能换吗？ / 尺寸差价怎么处理？
- id=44 recall@5=1.0 top1=0.2221 coverage=0.0 route=service manuals= question=请问你们的优惠券能用于所有商品吗？
- id=49 recall@5=1.0 top1=0.5776 coverage=1.0 route=service manuals= question=我购买的商品包装破损，商品出现损坏，而且快递员拒绝承认是运输问题，同时我已经签收商品，请问还能申请售后吗？
- id=55 recall@5=1.0 top1=0.5134 coverage=0.0 route=service manuals= question=我购买的商品，收到后发现商品的颜色和详情页描述的不一致，详情页是深红色，实际收到的是浅红色，而且商品有轻微的异味，想申请换货
- id=57 recall@5=1.0 top1=0.4961 coverage=0.1667 route=service manuals= question=我购买的商品，快递寄到后，我拆开包装发现商品损坏，但是快递员已经离开，没有当场验货，请问还能申请售后吗？
- id=58 recall@5=1.0 top1=0.5331 coverage=1.0 route=service manuals= question=我购买的商品，使用时发现商品的功能和详情页描述的不一致，详情页说支持无线充电，实际不支持，而且商品的续航时间也比描述的短很多，要求退货退款并赔偿，请问可以吗？",
- id=64 recall@5=1.0 top1=0.4372 coverage=0.0 route=manual manuals=吹风机手册|吹风机手册|吹风机手册 question=使用吹风机时，人员需要佩戴哪些防护装备？
- id=65 recall@5=1.0 top1=0.414 coverage=0.5 route=manual manuals=吹风机手册|吹风机手册|吹风机手册 question=操作吹风机时，人员需要注意哪些安全要点？
- id=66 recall@5=1.0 top1=0.4786 coverage=0.0 route=manual manuals=吹风机手册|吹风机手册|吹风机手册 question=使用吹风机时，如何调节化油器？
- id=67 recall@5=1.0 top1=0.4158 coverage=0.0 route=manual manuals=吹风机手册|吹风机手册|吹风机手册 question=吹风机冷机时，该如何启动？
- id=68 recall@5=1.0 top1=0.4252 coverage=0.0 route=manual manuals=吹风机手册|吹风机手册|吹风机手册 question=吹风机热机时，该如何启动？
- id=69 recall@5=1.0 top1=0.3693 coverage=0.0 route=manual manuals=吹风机手册|吹风机手册|吹风机手册 question=该如何关闭吹风机？
- id=70 recall@5=1.0 top1=0.2861 coverage=0.0 route=manual manuals=空调手册|空调手册|空调手册 question=空调的重要组成部件有哪些？
- id=71 recall@5=1.0 top1=0.5047 coverage=0.0 route=manual manuals=空调手册|空调手册|空调手册 question=如何找到空调遥控器的按键？