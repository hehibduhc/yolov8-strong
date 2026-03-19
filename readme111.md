# 工作日志

0301：今天改动是增加了spd_dcnv2.py文件，并且使用yaml的形式启用，替换backbone中的标准卷积吗，但是使用dcnv2需要安装额外的库，还没有搞定。
0302:装不好mmcv放弃可变形卷积的改动，
0303：新增mdka的改动，运行查看结果。新增四个mdka的改动，分别是基础c2fmdka，门控进阶c2fmdka，边界敏感+尺度自适应空洞c2fmdka三种，以不同yaml的形式启用，现在放在云平台上跑四组实验，明天看结果，继续寻找对应的模块中
0304:mdka的改动中，尺度自适应空洞卷积c2fmdka改进有效果,map提升0.7%,miou提升4%左右。新增sppf模块加入fcanet(频域卷积注意力)，但是效果不好，舍弃。
0305：新增sppf加入大核注意力模块lska，效果不好，新增sppf加入strippooling模块（轻量）效果也不好，继续寻找合适的模块
0309:新增replk轻量和全量模块，今天跑七个模块，mdka-sadiltion加sp三个模块的yaml，加上基础replk全量和轻量，加上mdka-sadiltion加replk全量和轻量结果
0310:sppf-replk-full的效果很好，但是和mdka-sadiltion叠加效果并不好，而且这两个map都没有什么提升，miou提升更多，新增HWD下采样模块取代backbone卷积，跑了看效果
0313:新增yolov8-seg-mcsta-p2.yaml，yolov8-seg-agm.yaml，yolov8-seg-ciepool-afp-secbam.yaml, 跑实验中
0314:以上四个改进没有用，特别是mcsta，失去了本身的检测能力。目前有改进的还是，mdka-sadilition,replk-full，但是这俩组合会下降，并且这两个改进map都没什么提高，我现在针对这两个改进更换了一个数据集（更复杂），打算看一看效果。新增了SPD下采样模块，替换13卷积，和论文中的一样，还没更新到云平台，等明天训练看看效果

## 0315工作日志
（1）采用论文同样的数据集（paper），使用baseline:yolov8-seg,使用yolov8-seg-mdka-sadilation，yolov8-seg-sppf-replk-full这两个在简单数据集上表现优异的模块进行训练，得到的结果是yolov8-seg-sppf-replk-full在P,R,MAP三个指标上均远超baseline,但是miou接近。而yolov8-seg-mdka-sadilation在p,R,MAP上均有3%的提升，但是miou下降了2%，模块spd13的效果也很不错，模块spd13+replk的效果也可以
（2）将spd13+replksppf+mdka-sadilation三个模块组合的效果还不错，准备跑多个seed看看平均结果，计划跑42，43，44这三个种子，42已经跑过了，所以跑42，43两个种子，每个种子有八组实验，预计明天可以跑完，如果效果理想的话就想想怎么解释实验结果，然后在数据集和任务上下功夫改进

## 0316工作日志
（1）315的工作日志的无效的，因为训练用的是paper数据集，验证却用的是另外一个简单数据集
（2）尝试使用四类裂缝的数据集进行训练，看看效果如何，结果是改进模型效果均不如原始模型

## 0317工作日志
（1）今天让codex自己找改进方向，把我代码改了将mdka-sadilation和gated结合起来，效果最好，miou达到了88,其他不行

# 工作文件

新增了一个工具，在tools-excel0313中，用于按照指定指标提取文件夹中result.csv的最好轮次结果进行比较