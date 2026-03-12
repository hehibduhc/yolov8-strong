0301：今天改动是增加了spd_dcnv2.py文件，并且使用yaml的形式启用，替换backbone中的标准卷积吗，但是使用dcnv2需要安装额外的库，还没有搞定。
0302:装不好mmcv放弃可变形卷积的改动，
0303：新增mdka的改动，运行查看结果。新增四个mdka的改动，分别是基础c2fmdka，门控进阶c2fmdka，边界敏感+尺度自适应空洞c2fmdka三种，以不同yaml的形式启用，现在放在云平台上跑四组实验，明天看结果，继续寻找对应的模块中
0304:mdka的改动中，尺度自适应空洞卷积c2fmdka改进有效果,map提升0.7%,miou提升4%左右。新增sppf模块加入fcanet(频域卷积注意力)，但是效果不好，舍弃。
0305：新增sppf加入大核注意力模块lska，效果不好，新增sppf加入strippooling模块（轻量）效果也不好，继续寻找合适的模块
0309:新增replk轻量和全量模块，今天跑七个模块，mdka-sadiltion加sp三个模块的yaml，加上基础replk全量和轻量，加上mdka-sadiltion加replk全量和轻量结果
0310:sppf-replk-full的效果很好，但是和mdka-sadiltion叠加效果并不好，而且这两个map都没有什么提升，miou提升更多，新增HWD下采样模块取代backbone卷积，跑了看效果